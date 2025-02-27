# %%
# %%
import os
import json
import csv
import pandas as pd
import torch
import torch.cuda as cuda
from datasets import Dataset as HFDataset
from datasets import load_dataset
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import torch
from scipy.stats import entropy
import argparse
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset, DataLoader
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    LlamaModel, 
)
from torch.utils.data import Dataset, DataLoader
from fsd_utils import FuzzyAssistedCandidateGenerator, FSDAutoModelForCausalLM, FSDLlamaForCausalLM, FSDGemma2ForCausalLM

# %%
dataset = load_dataset('tau/commonsense_qa')    

# %%
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device3 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

print(f"device0: {device0} device1: {device1}, device2: {device2}, device3: {device3}")



# %%

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Script Configuration')
    

    # Add arguments
    parser.add_argument('--small_model_id', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help='Backoff threshold')
    parser.add_argument('--large_model_id', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct", help='Backoff threshold')
    parser.add_argument('--fsd_div_threshold', type=float, default=None, help='Backoff threshold')
    parser.add_argument('--num_evals', type=int, default=1, help='Number of evaluations')
    parser.add_argument('--no_do_sample', action='store_false', dest='do_sample', help='Do not use cache')
    parser.add_argument('--exp', type=str, default=f'', help='Experiment name')
    parser.add_argument('--fsd_div_type', type=str, default='kl_div', help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='Positive probability threshold')
    parser.add_argument('--top_p', type=float, default=0.9, help='Positive probability threshold')
    parser.add_argument('--top_k', type=int, default=50, help='Positive probability threshold')

    args, unknown = parser.parse_known_args()

small_model_id = args.small_model_id
large_model_id = args.large_model_id
exp = args.exp
fsd_div_threshold = args.fsd_div_threshold
fsd_div_type = args.fsd_div_type
num_evals = args.num_evals
do_sample = args.do_sample
fsd_div_type = args.fsd_div_type
temperature = args.temperature
top_p = args.top_p
top_k = args.top_k

print(f"loaded arguments - small_model_id: {small_model_id}, large_model_id: {large_model_id}, fsd_div_threshold: {fsd_div_threshold}, num_evals: {num_evals}, do_sample: {do_sample}, fsd_div_type: {fsd_div_type}, temperature: {temperature}, top_p: {top_p}")

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
small_model = FSDAutoModelForCausalLM.from_pretrained(small_model_id, torch_dtype=torch.bfloat16).to(device0)
model = FSDAutoModelForCausalLM.from_pretrained(small_model_id, torch_dtype=torch.bfloat16, device_map='auto')

def eval_commonsenseqa(model, small_model, fsd_div_threshold, fsd_div_type, dataset, tokenizer=tokenizer, device0=device0, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature, eval_num=0, exp=exp):
    model_outputs = {'data': []}
    num_words_lengths = []
    backoff_percentages = []
    total_generated_tokens = []

    with torch.no_grad():
        for i, item in tqdm(enumerate(dataset), total=len(dataset)):
            question = item['question']
            question_id = item['id']
            question_concept = item['question_concept']
            choices = item['choices']
            answer_key = item['answerKey']
            text_choices = choices['text']
            
            #-------------------------NOTE--------------------------------------------
            #The initial prompt was built for Llama3.1 models. We then adjust this prompt for Gemma2 and Qwen2.5 models to use their respective prompt formats. 
            #---------------------------------------------------------------------
            
            formatted_question = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful and knowledgable assistant that answer a series of multiple-choice question that require common sense reasoning. To answer each question, consider the context and use your understanding of the world. Analyze the question and options carefully, then select the most appropriate answer. <|eot_id|>
            
            <|start_header_id|>user<|end_header_id|> You will be given a multiple-choice question that requires common sense reasoning. For each question, select one of the given answers options that most accurately answers the question. Provide a detailed chain of thought that explains your reasoning leading to the answer.' Only answer with your explanation and your answer, ending with 'Therefore, the answer is ANSWER_LETTER.', following the examples below:
            
            Example 1:
            Question: Sammy wanted to go to where the people were. Where might he go?

            Choices: A) race track, B) populated areas, C) the desert, D) apartment, E) roadblock

            Answer: Sammy wants to find a place with many people. A race track might have people, but only during events. An apartment houses people but isn't generally crowded. A roadblock is not a gathering place. The desert is generally uninhabited. Populated areas, by definition, have many people.
            Therefore, the answer is B.

            Example 2:
            Question: To locate a choker not located in a jewelry box or boutique where would you go?

            Choices: A) jewelry store, B) neck, C) jewelry box, D) jewelry box, E) boutique

            Answer: A choker is a type of necklace worn around the neck. Since the question specifies the choker is not in a jewelry box or boutique, the next likely place to find it would be around someone's neck, as that is where it is typically worn.
            Therefore, the answer is B.

            Example 3:
            Question: Google Maps and other highway and street GPS services have replaced what?

            Choices: A) united states, B) mexico, C) countryside, D) atlas, E) oceans

            Answer: GPS services provide detailed maps and directions, functions that were traditionally served by an atlas. The other choices do not serve the same purpose as GPS.
            Therefore, the answer is D.

            Example 4:
            Question: The fox walked from the city into the forest, what was it looking for?

            Choices: A) pretty flowers, B) hen house, C) natural habitat, D) storybook, E) dense forest

            Answer: A fox moving from a city to a forest is likely seeking its natural habitat, where it can find food and shelter. The other options are either too specific or unrelated to a fox's natural behavior.
            Therefore, the answer is C.

            Example 5:
            Question: What home entertainment equipment requires cable?

            Choices: A) radio shack, B) substation, C) cabinet, D) television, E) desk

            Answer: Televisions often require cable for access to a wide range of channels. The other options are not home entertainment equipment that use cables in this context.
            Therefore, the answer is D. 
            
            Question: {question}
            
            Choices: A) {text_choices[0]}, B) {text_choices[1]}, C) {text_choices[2]}, D) {text_choices[3]}, E) {text_choices[4]} <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
            Answer: '''
            
            llama_special_tokens = ['<|start_header_id|>user<end_header_id|>', '<|start_header_id|>assistant<|end_header_id|>', '<|eot_id|>', '<end_header_id|>', '<|start_header_id|>']
            
            if 'gemma' in small_model_id:
                formatted_question = formatted_question.replace('<|start_header_id|>user<|end_header_id|>', '<start_of_turn>user')
                formatted_question = formatted_question.replace('<|eot_id|>', '<end_of_turn>\n')
                formatted_question = formatted_question.replace('<|start_header_id|>assistant<|end_header_id|>', '<start_of_turn>model')
                
                for token in llama_special_tokens:
                    formatted_question = formatted_question.replace(token, '')
                
            elif 'Qwen' in small_model_id:
                #remove all llama special tokens from prompt
                for token in llama_special_tokens:
                    formatted_question = formatted_question.replace(token, '')
            
            question_tokenized = tokenizer(formatted_question, return_tensors='pt').to(device0)
            
            if fsd_div_threshold == None:
                outputs = model.generate(**question_tokenized, max_length=question_tokenized['input_ids'].shape[1] + 512, do_sample=do_sample, assistant_model=small_model, temperature=temperature, top_k=top_k, top_p=top_p)
            else:
                outputs = model.generate(**question_tokenized, max_length=question_tokenized['input_ids'].shape[1] + 512, do_sample=do_sample, assistant_model=small_model, temperature=temperature, top_k=top_k, top_p=top_p, fsd_div_threshold=fsd_div_threshold, fsd_div_type=fsd_div_type)
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer_tokens = outputs[0][question_tokenized['input_ids'].shape[1]:]
            num_words_lengths.append(outputs.shape[1])
            
            total_generated_tokens.append(generated_answer_tokens.shape[-1])
            generated_answer = tokenizer.decode(generated_answer_tokens, skip_special_tokens=True)
            answer_start = generated_text.find(formatted_question) + len(formatted_question)
            answer = generated_text[answer_start - 1:].strip()
            
            model_outputs['data'].append({
                'question': question,
                'question_id': question_id,
                'question_concept': question_concept,
                'choices': choices,
                'answer_key': answer_key,
                'output': generated_answer,
                'output_tokens': generated_answer_tokens.tolist(),
                'full_output_tokens': outputs[0].tolist(),
            })
            if i == 5:
                break
    return model_outputs, total_generated_tokens

import re

def extract_latest_answer(response):
    # Split the response into sentences
    sentences = response.split('. ')
    
    # Reverse the list of sentences so we start from the end
    sentences.reverse()
    
    # Compile a regular expression pattern for the answer structure
    pattern = re.compile(r"Therefore, the answer is ([A-Za-z])")
    
    # Loop over the sentences in reverse order
    for sentence in sentences:
        # Search for the pattern in the sentence
        match = pattern.search(sentence)
        
        # If a match is found, return the letter of the predicted answer
        if match:
            return match.group(1)
    
    # If no match is found, return None
    print(f"NO MATCH FOUND: {response}")
    return None

# %%
def calc_accuracy(results):
    correct = 0
    total = 0
    parse_error = 0
    
    for result in results['data']:
        output = result['output']
        extracted_answer = extract_latest_answer(output)
        answer_key = result['answer_key']
        
        if extracted_answer is None:
            parse_error += 1
        
        # print(f"Extracted Answer: {extracted_answer}, Answer Key: {answer_key}")
        
        if extracted_answer == answer_key:
            correct += 1
        
        total += 1
    
    return correct/total, parse_error





accuracies = []
parse_errors = []
ts = []
n_matches = []
candidate_lengths = []
backoff_token_percentage = []

for i in range(num_evals):
    print(f"starting evaluation {i}")
    starting_time = time.time()
    eval_outputs, total_generated_tokens = eval_commonsenseqa(model, small_model, fsd_div_threshold, fsd_div_type, dataset['validation'], tokenizer=tokenizer, device0=device0, eval_num=i)
    elapsed_time = time.time() - starting_time

    accuracy, parse_error = calc_accuracy(eval_outputs)
    accuracies.append(accuracy) 
    parse_errors.append(parse_error)

    if fsd_div_threshold is None:
        print(f"Evaluating SD")
        print(f"Accuracy on evaluation {i}: {accuracy}")
    else:
        print(f"Evaluating FSD with threshold: {fsd_div_threshold} and div type: {fsd_div_type}")
        print(f"Accuracy on evaluation {i}: {accuracy}")
        print(f"tokens per second: {sum(total_generated_tokens)/elapsed_time}")
        print(f"average n_matches: {sum(small_model.n_matches_list)/len(small_model.n_matches_list)}")
        print(f"average candidate_length: {sum(small_model.candidate_sequences_list)/len(small_model.candidate_sequences_list)}")
        print(f"percetange of MD tokens: {sum(small_model.n_matches_list)/sum(small_model.totals_list)}")
