# FSD implementation from "Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff"

This is the official fuzzy speculative decoding implementation used to run experiments in the paper "Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff".

## How to use:

We designed our implementation on top of the huggingface `transformers` library for easy use. As such, our implementation modifies the default `model.generate` assisted decoding functionality to also allow for fuzzy speculative decoding in addition to regular speculative decoding. 

To use our implementation, follow the steps below: 
1. Install `transformers==4.44`
2. Our modified `model.generate` functionality can easily be accessed via custom `ForCausalLM` classes. Currently, we have implemented custom classes for all models tested in our paper, namely `LlamaForCausalLM`, `Gemma2ForCausalLM` and `Qwen2ForCausalLM`. These our custom version of these three model classes can be accessed through a custom class `FSDAutoModelForCausalLM`. Thus, to initialize:

```python
from transformers
from fsd.fsd_utils import FSDAutoModelForCausalLM

small_model_id = 'google/gemma-2-2b-it'
large_model_id = 'google/gemma-2-27b-it'

small_tokenizer = AutoTokenizer.from_pretrained(small_model_id)
small_model = FSDAutoModelForCausalLM.from_pretrained(small_model_id, torch_dtype=torch.bfloat16).to(device0)
large_model = FSDAutoModelForCausalLM.from_pretrained(large_model_id, torch_dtype=torch.bfloat16, device_map='auto')
```

5. Use FSD as you would use regular speculative decoding by passing as assistant model to the target model's `model.generate` call. Set the divergence type with the `fsd_div_type` parameter (defaults to JS divergence), and the div threshold with the `fsd_div_threshold` parameter. *Whether FSD or traditional SD is run depends on whether `fsd_div_threshold` is set to a value - if this parameter is not passed into `model.generate`, regular SD will run*

```python 
input_text = "Write me an essay about the massive risks of climate change."
input_ids = small_tokenizer(input_text, return_tensors='pt').to(device0)

output = large_model.generate(**input_ids, assistant_model=small_model, fsd_div_threshold=0.4, fsd_div_type='js_div', max_new_tokens=250)

print(f"output: {output}")
print(f"output: {small_tokenizer.decode(output[0])}")
```

The divergence options are KL divergence (`kl_div`), JS divergence (`js_div`), TV distance (`tv_div`), as well as top-K and top-P variants of these three divergences (`top_k_kl_div`, `top_k_js_div`, `top_k_tv_div`). Note that the `fsd_div_threshold` is highly dependent on which divergence type is being used. 

## Example usage
`csqa_eval_example.py` gives an example of how we evaluated FSD (this file is a cleaned up version of the exact code we used). This code can easily be modified to evaluate new datasets. 

To use, simply pass the desired arguments. For example: 

```bash
python csqa_eval_example.py --small_model_id "google/gemma-2-2b-it" --large_model_id "google/gemma-2-27b-it" --fsd_div_threshold 0.4 --fsd_div_type "js_div" --num_evals 5
```

