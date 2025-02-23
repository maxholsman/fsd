# FSD implementation from "Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff"

This is the official fuzzy speculative decoding implementation used to run experiments in the paper "Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff".

## How to use:

We designed our implementation on top of the huggingface `transformers` library for easy use. As such, our implementation modifies the default `model.generate` assisted decoding functionality to also allow for fuzzy speculative decoding in addition to regular speculative decoding. 

To use our implementation, follow the steps below: 
1. Install `transformers` version 4.44
2. Install `fsd_utils.py`, which is a modified version of `transformers/generation/utils.py`, with your method of choice.
3. Import the necessary `transformers` classes, as well as `FuzzyGenerationMixin` from `fsd_utils.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fsd.fsd_utils import FuzzyGenerationMixin
```

4. When defining the target and draft models, reroute the use of the default `GenerationMixin` with:

```python
small_tokenizer = AutoTokenizer.from_pretrained(small_model_id)
small_model = AutoModelForCausalLM.from_pretrained(small_model_id, torch_dtype=torch.bfloat16).to(device0)
large_model = AutoModelForCausalLM.from_pretrained(large_model_id, torch_dtype=torch.bfloat16, device_map='auto')

def patch_generation(model):
    """Patches model.generate functions to use modified code from FuzzyGenerationMixin"""
    
    # Preserve model-specific `prepare_inputs_for_generation`
    if not hasattr(model, "prepare_inputs_for_generation") or model.prepare_inputs_for_generation.__func__ == fsd_utils.FuzzyGenerationMixin.prepare_inputs_for_generation:
        # Get method from original model class (e.g., GemmaForCausalLM)
        for base_class in model.__class__.__mro__:
            if "prepare_inputs_for_generation" in base_class.__dict__:
                model.prepare_inputs_for_generation = base_class.prepare_inputs_for_generation.__get__(model, model.__class__)
                break  # Stop once we find it
    
    # Override generation-related methods with fsd_utils
    model.generate = fsd_utils.FuzzyGenerationMixin.generate.__get__(model, model.__class__)
    model._backoff_assisted_decoding = fsd_utils.FuzzyGenerationMixin._backoff_assisted_decoding.__get__(model, model.__class__)
    model._get_candidate_generator = fsd_utils.FuzzyGenerationMixin._get_candidate_generator.__get__(model, model.__class__)
    model._update_model_kwargs_for_generation = fsd_utils.FuzzyGenerationMixin._update_model_kwargs_for_generation.__get__(model, model.__class__)


# Apply the patch to both models
patch_generation(small_model)
patch_generation(large_model)
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


