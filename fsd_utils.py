import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, LlamaForCausalLM, Gemma2ForCausalLM, Qwen2ForCausalLM, AutoConfig

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HQQQuantizedCache,
    HybridCache,
    MambaCache,
    OffloadedCache,
    QuantizedCacheConfig,
    QuantoQuantizedCache,
    SlidingWindowCache,
    StaticCache,
)

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils import (
    ModelOutput,
    is_accelerate_available,
    is_hqq_available,
    is_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
    # FuzzyAssistedCandidateGenerator,
    BackoffCandidateGenerator,
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
    WatermarkLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

from transformers.generation.utils import GenerateOutput, GenerationMixin, GenerateBeamEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateBeamDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateNonBeamOutput

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
    "sliding_window": SlidingWindowCache,
    "hybrid": HybridCache,
    "mamba": MambaCache,
}
QUANT_BACKEND_CLASSES_MAPPING = {"quanto": QuantoQuantizedCache, "HQQ": HQQQuantizedCache}

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.candidate_generator import CandidateGenerator


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.configuration_utils import GenerationConfig
    
class AssistedCandidateGenerator(CandidateGenerator): # VERY slightly changed AssistedCandidateGenerator class to return both candidates AND logits (as opposed to just candidates)
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """
    
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        # Make sure all data at the same device as assistant model
        device = assistant_model.device
        input_ids = input_ids.to(device)
        if inputs_tensor is not None:
            inputs_tensor = inputs_tensor.to(device)
        
        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
        self.num_assistant_tokens = 5
        
        # Set eos in assistant same as in target model
        self.assistant_model.generation_config.eos_token_id = generation_config.eos_token_id
        self.assistant_model.generation_config.num_assistant_tokens_schedule = 'None'
        
        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs", "past_key_values"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name, assistant_model.generation_config
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True
        self.generation_config.output_logits = True

        # Disable sampling -- this implementation of assisted generation/speculative decoding uses the assistant
        # greedily to maximize matches. Disables sampling-related flags to prevent warnings
        
        # print(f"self.generation_config.do_sample: {self.generation_config.do_sample}")
        self.generation_config.do_sample = False # Change this back to False
        for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k", "epsilon_cutoff", "eta_cutoff"):
            setattr(self.generation_config, attr, None)
        
        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None
        for processor in self.logits_processor:
            if isinstance(processor, MinLengthLogitsProcessor):
                raise ValueError(
                    "Passing `MinLengthLogitsProcessor` when using `assisted_generation is disabled. "
                    "Please pass in `min_length` into `.generate()` instead"
                )

        # We need to roll back the cache in assisted generation, only DynamicCache is supported
        self.generation_config.cache_implementation = None

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        `A`rgs:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        
        """
        
        input_ids = input_ids.to(self.assistant_model.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        # print(f"max_new_tokens: {max_new_tokens} - num assistant tokens: {self.num_assistant_tokens} - max_length: {self.generation_config.max_length} - new cur len: {new_cur_len} - main_model: {self.main_model_min_length}")
        if max_new_tokens == 0:
            return input_ids, None, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length 
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }
        
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_logits_unprocessed = torch.stack(assistant_output.logits, dim=1)
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits, candidate_logits_unprocessed

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)

    
class FuzzyAssistedCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """
    
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        # Make sure all data at the same device as assistant model
        device = assistant_model.device
        input_ids = input_ids.to(device)
        if inputs_tensor is not None:
            inputs_tensor = inputs_tensor.to(device)
        
        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
        self.num_assistant_tokens = 5
        
        # Set eos in assistant same as in target model
        self.assistant_model.generation_config.eos_token_id = generation_config.eos_token_id
        self.assistant_model.generation_config.num_assistant_tokens_schedule = 'None'
        
        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs", "past_key_values"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name, assistant_model.generation_config
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True
        self.generation_config.output_logits = True

        # Disable sampling -- this implementation of assisted generation/speculative decoding uses the assistant
        # greedily to maximize matches. Disables sampling-related flags to prevent warnings
        
        # print(f"self.generation_config.do_sample: {self.generation_config.do_sample}")
        self.generation_config.do_sample = False # Change this back to False
        for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k", "epsilon_cutoff", "eta_cutoff"):
            setattr(self.generation_config, attr, None)
        
        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None
        for processor in self.logits_processor:
            if isinstance(processor, MinLengthLogitsProcessor):
                raise ValueError(
                    "Passing `MinLengthLogitsProcessor` when using `assisted_generation is disabled. "
                    "Please pass in `min_length` into `.generate()` instead"
                )

        # We need to roll back the cache in assisted generation, only DynamicCache is supported
        self.generation_config.cache_implementation = None

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        `A`rgs:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        
        """
        
        input_ids = input_ids.to(self.assistant_model.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        # print(f"max_new_tokens: {max_new_tokens} - num assistant tokens: {self.num_assistant_tokens} - max_length: {self.generation_config.max_length} - new cur len: {new_cur_len} - main_model: {self.main_model_min_length}")
        if max_new_tokens == 0:
            return input_ids, None, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length 
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }
        
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_logits_unprocessed = torch.stack(assistant_output.logits, dim=1)
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits, candidate_logits_unprocessed

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)
                
class FuzzyGenerationMixin (GenerationMixin):

    def _get_candidate_generator(
        self,
        generation_config: GenerationConfig,
        input_ids: torch.LongTensor,
        inputs_tensor: torch.Tensor,
        assistant_model: "PreTrainedModel",
        logits_processor: LogitsProcessorList,
        fsd: bool,
        model_kwargs: Dict,
    ) -> CandidateGenerator:
        """
        Returns the candidate generator to be used in `assisted_generation`
        """
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                eos_token_id=generation_config._eos_token_tensor,
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
                max_length=generation_config.max_length,
            )
        else:
            
            if fsd:
                # print(f"setting candidate generator to FuzzyAssistedCandidateGenerator")
                candidate_generator = FuzzyAssistedCandidateGenerator(
                    input_ids=input_ids,
                    assistant_model=assistant_model,
                    generation_config=generation_config,
                    model_kwargs=model_kwargs,
                    inputs_tensor=inputs_tensor,
                    logits_processor=logits_processor,
                )
            else:
                candidate_generator = AssistedCandidateGenerator(
                    input_ids=input_ids,
                    assistant_model=assistant_model,
                    generation_config=generation_config,
                    model_kwargs=model_kwargs,
                    inputs_tensor=inputs_tensor,
                    logits_processor=logits_processor,
                )
                
        return candidate_generator

    @classmethod
    def can_generate(cls) -> bool:
        return True  # Force models using this mixin to be considered valid for generation

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        fsd_div_threshold: Optional[float] = None,
        fsd_div_type: Optional[str] = None,
        fsd_div_logit_processor: Optional[LogitsProcessorList] = None,
        fsd_tracking: Optional[bool] = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        kwargs_has_attention_mask_large = model_kwargs.get("attention_mask_large", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        
        # print(f"cache_implementation: {generation_config.cache_implementation}")
        use_dynamic_cache_by_default = False
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"

        # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
        # which is only supported in dynamic caches atm
        if (
            assistant_model is not None
            and generation_config.cache_implementation is not None
            and self._supports_default_dynamic_cache()
        ):
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        if (model_kwargs.get(cache_name) is not None) and is_torchdynamo_compiling():
            raise ValueError(
                "Passing `past_key_values` is not supported when compiling `model.generate` with torch.compile -- you "
                "may get incorrect outputs. Please compile `model.forward` only or use the `cache_implementation` "
                "input argument."
            )
        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    max_batch_size=generation_config.num_beams * generation_config.num_return_sequences * batch_size,
                    max_cache_len=generation_config.max_length,
                    device=device,
                    model_kwargs=model_kwargs,
                )
                # SHOULD PROBABLY ADD BACKOFF CACHE INITIALIZATION HERE
                
                # print(f"cache after get cache: {model_kwargs.get(cache_name, None)}")
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs[cache_name] = cache_class(cache_config)
            elif generation_config.cache_implementation == "offloaded":
                model_kwargs[cache_name] = OffloadedCache()
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                # print(f"found where cache is initialized")
                # print(f"cache initialized")
                model_kwargs[cache_name] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                if hasattr(self, 'backoff_model'):
                    # print(f"initializing backoff cache correctly")
                    model_kwargs["past_backoff_key_values"] = (
                        DynamicCache()
                        if not requires_cross_attention_cache
                        else EncoderDecoderCache(DynamicCache(), DynamicCache())
                    )
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)
        
        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )
        
        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            
            # if not(hasattr(assistant_model, 'div_threshold')):
            print(not(hasattr(assistant_model, 'div_threshold')))
            if fsd_div_threshold is None:
                if generation_config.num_return_sequences > 1:
                    raise ValueError(
                        "num_return_sequences has to be 1 when doing assisted generate, "
                        f"but is {generation_config.num_return_sequences}."
                    )
                if batch_size > 1:
                    raise ValueError("assisted generate is only supported for batch_size = 1")
                if not model_kwargs["use_cache"]:
                    raise ValueError("assisted generate requires `use_cache=True`")
                if generation_config.cache_implementation == "static":
                    raise ValueError("assisted generate is not supported with `static_cache`")
                if self._is_stateful:
                    # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                    # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                    raise ValueError(
                        f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                    )

                # 11. Get the candidate generator, given the parameterization
                candidate_generator = self._get_candidate_generator(
                    generation_config=generation_config,
                    input_ids=input_ids,
                    inputs_tensor=inputs_tensor,
                    assistant_model=assistant_model,
                    logits_processor=logits_processor,
                    model_kwargs=model_kwargs,
                    fsd=False
                )

                # 12. prepare logits warper (if `do_sample` is `True`)
                prepared_logits_warper = (
                    self._get_logits_warper(
                        generation_config,
                        device=input_ids.device,
                    )
                    if generation_config.do_sample
                    else None
                )

                # 13. run assisted generate
                result = self._assisted_decoding(
                    input_ids,
                    candidate_generator=candidate_generator,
                    logits_processor=prepared_logits_processor,
                    logits_warper=prepared_logits_warper,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    synced_gpus=synced_gpus,
                    streamer=streamer,
                    **model_kwargs,
                )
            else: # if fsd_div_threshold is not None -> FSD
                assistant_model.div_threshold = fsd_div_threshold
                
                if generation_config.num_return_sequences > 1:
                    raise ValueError(
                        "num_return_sequences has to be 1 when doing assisted generate, "
                        f"but is {generation_config.num_return_sequences}."
                    )
                
                if batch_size > 1:
                    raise ValueError("assisted generate is only supported for batch_size = 1")
                if not model_kwargs["use_cache"]:
                    raise ValueError("assisted generate requires `use_cache=True`")
                if generation_config.cache_implementation == "static":
                    raise ValueError("assisted generate is not supported with `static_cache`")
                if self._is_stateful:
                    # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                    # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                    raise ValueError(
                        f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                    )

                # 11. Get the candidate generator, given the parameterization
                candidate_generator = self._get_candidate_generator(
                    generation_config=generation_config,
                    input_ids=input_ids,
                    inputs_tensor=inputs_tensor,
                    assistant_model=assistant_model,
                    logits_processor=logits_processor,
                    model_kwargs=model_kwargs,
                    fsd=True,
                )

                # 12. prepare logits warper (if `do_sample` is `True`)
                prepared_logits_warper = (
                    self._get_logits_warper(
                        generation_config,
                        device=input_ids.device,
                    )
                    if generation_config.do_sample
                    else None
                )

                # 13. run assisted generate
                result = self._backoff_assisted_decoding(
                    input_ids,
                    candidate_generator=candidate_generator,
                    logits_processor=prepared_logits_processor,
                    logits_warper=prepared_logits_warper,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    synced_gpus=synced_gpus,
                    streamer=streamer,
                    fsd_div_type=fsd_div_type,
                    fsd_div_logit_processor=fsd_div_logit_processor,
                    **model_kwargs,
                )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
                )
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )
            #     print(f"cache is None before expand inputs")
            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: CandidateGenerator,
        logits_processor: LogitsProcessorList,
        logits_warper: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only used if sampling is active.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        do_sample = logits_warper is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # This is needed if return_dict_in_generate is True
        start_from_empty_dynamic_cache = False
        past_key_values = model_kwargs.get("past_key_values", None)
        if isinstance(past_key_values, DynamicCache) or (
            isinstance(past_key_values, EncoderDecoderCache)
            and isinstance(past_key_values.self_attention_cache, DynamicCache)
        ):
            if len(past_key_values) == 0:
                start_from_empty_dynamic_cache = True

        this_peer_finished = False
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            overall_start_time = time.time()
            cur_len = input_ids.shape[-1]

            #  1. Fetch candidate sequences from a `CandidateGenerator`
            start_time = time.time()
            candidate_input_ids, candidate_logits, candidate_logits_unprocessed = candidate_generator.get_candidates(input_ids)
            # print(f"candidate_logits: {candidate_logits}")
            if candidate_logits is not None:
                candidate_logits = candidate_logits.to(self.device)

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            is_done_candidate = stopping_criteria(candidate_input_ids, None)
            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )

            model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = candidate_length + 1

            # 2.2. Run a forward pass on the candidate sequence
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            start_time = time.time()

            outputs = self(**model_inputs)
            
            start_time = time.time()
            # 2.3. Process the new logits - will likely move this into speculative_decoding_sampling when I write this
            new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            next_token_logits = new_logits.clone()
            
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
            if do_sample and len(logits_warper) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
            logit_processing_time = time.time() - start_time

            # 3. Select the accepted tokens. There are two possible cases:
            # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
            # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
            
            if do_sample and candidate_logits is not None:
                start_time = time.time()
                valid_tokens, n_matches, correction_term, is_accepted, acceptance_time, spec_sampling_time = _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate,
                )
                spec_sampling_time = time.time() - start_time
                start_time = time.time()
                
                # candidate_generator.assistant_model.acceptance_list.extend(is_accepted.int().view(-1).tolist())
                
                start_time = time.time()
                if hasattr(candidate_generator, "assistant_model"):
                    if hasattr(candidate_generator.assistant_model, "n_matches_list"):
                        candidate_generator.assistant_model.n_matches_list.append(n_matches + correction_term)
                        candidate_generator.assistant_model.totals_list.append(valid_tokens.shape[-1])
                        # if n_matches == candidate_length:
                        #     candidate_generator.assistant_model.forced_ml_generations += 1
                        candidate_generator.assistant_model.n_discarded_list.append(candidate_input_ids.shape[-1] - n_matches)
                        candidate_generator.assistant_model.candidate_sequences_list.append(candidate_length)
            # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
            # original model logits with the candidate tokens. We can keep the candidate tokens until the first
            # mismatch, or until the max length is reached.
            else:
                if do_sample:
                    probs = new_logits.softmax(dim=-1)
                    selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                else:
                    selected_tokens = new_logits.argmax(dim=-1)

                candidate_new_tokens = candidate_input_ids[:, cur_len:]
                n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

                # Ensure we don't generate beyond max_len or an EOS token
                if is_done_candidate and n_matches == candidate_length:
                    n_matches -= 1
                valid_tokens = selected_tokens[:, : n_matches + 1]

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            if streamer is not None:
                streamer.put(valid_tokens.cpu())
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

            # 5. Update the candidate generation strategy if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
                if output_logits:
                    raw_logits += (next_token_logits,)

                if "past_key_values" not in model_kwargs or start_from_empty_dynamic_cache:
                    added_len = new_cur_len
                    # set it to false for other iterations
                    start_from_empty_dynamic_cache = False
                else:
                    added_len = n_matches + 1

                if output_attentions:
                    if self.config.is_encoder_decoder:
                        cross_attentions = _split_model_outputs(
                            cross_attentions, outputs.cross_attentions, cur_len, added_len
                        )
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.decoder_attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                    else:
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                if output_hidden_states:
                    if self.config.is_encoder_decoder:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                        )
                    else:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                        )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=n_matches + 1,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if (
            hasattr(candidate_generator, "assistant_model")
            and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
                candidate_generator.num_assistant_tokens
            )
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
    
    def _backoff_assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: CandidateGenerator,
        logits_processor: LogitsProcessorList,
        logits_warper: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        fsd_div_type: str="js_div",
        div_logit_processor: Optional[LogitsProcessorList]=[],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only used if sampling is active.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        do_sample = logits_warper is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # This is needed if return_dict_in_generate is True
        start_from_empty_dynamic_cache = False
        past_key_values = model_kwargs.get("past_key_values", None)
        if isinstance(past_key_values, DynamicCache) or (
            isinstance(past_key_values, EncoderDecoderCache)
            and isinstance(past_key_values.self_attention_cache, DynamicCache)
        ):
            if len(past_key_values) == 0:
                start_from_empty_dynamic_cache = True

        this_peer_finished = False
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            overall_start_time = time.time()
            cur_len = input_ids.shape[-1]
            start_time = time.time()
            #  1. Fetch candidate sequences from a `CandidateGenerator`

            candidate_input_ids, candidate_logits, candidate_logits_unprocessed = candidate_generator.get_candidates(input_ids)
            # print(f"candidate logits: {candidate_logits}")
            # print(f"candidate logits unprocessed: {candidate_logits_unprocessed}")
            if candidate_logits is not None:
                candidate_logits = candidate_logits.to(self.device)

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            is_done_candidate = stopping_criteria(candidate_input_ids, None) # might have to check this, but seems fine
            if candidate_length == 5:
                if not(hasattr(candidate_generator.assistant_model, "candidate_generator_times")):
                    candidate_generator.assistant_model.candidate_generator_times = []
                candidate_generator.assistant_model.candidate_generator_times.append((time.time() - start_time)/candidate_length)
    
            # print(f"candidate_generator_time: {(time.time() - start_time)/candidate_length}")
            start_time = time.time()

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )

            model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = candidate_length + 1

            # 2.2. Run a forward pass on the candidate sequence
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            
            outputs = self(**model_inputs)
            start_time = time.time()
            # 2.3. Process the new logits - will likely move this into speculative_decoding_sampling when I write this
            new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            new_logits_unprocessed = new_logits.clone()
            
            candidate_generator.assistant_model.div_logit_processor = div_logit_processor
            
            if len(candidate_generator.assistant_model.div_logit_processor) != 0:
                next_token_logits = new_logits.clone()
                print(f"processing div logits...")
            else:
                next_token_logits = new_logits_unprocessed
            
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
                    if i < candidate_length:
                        candidate_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], candidate_logits[:, i, :])
            div_logits_processor = LogitsProcessorList()
            if do_sample and len(logits_warper) > 0:
                # epsilon = 1e-10
                if 'temperature' in candidate_generator.assistant_model.div_logit_processor:
                    div_logits_processor.append(logits_warper[0])
                if 'top_k' in candidate_generator.assistant_model.div_logit_processor:
                    div_logits_processor.append(logits_warper[1])
                if 'top_p' in candidate_generator.assistant_model.div_logit_processor:
                    div_logits_processor.append(logits_warper[2])
                
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
                    if len(div_logits_processor) > 0:
                        print(f"processing new logits...")
                        new_logits_unprocessed[:, i, :] = div_logits_processor(candidate_input_ids[:, : cur_len + i], new_logits_unprocessed[:, i, :])
                    if i < candidate_length:
                        # print(f"candidate_logits processing...")
                        candidate_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], candidate_logits[:, i, :])
                        if len(div_logits_processor) > 0:
                            print(f"processing candidate logits...")
                            candidate_logits_unprocessed[:, i, :] = div_logits_processor(candidate_input_ids[:, : cur_len + i], candidate_logits_unprocessed[:, i, :])
            
            logit_processing_time = time.time() - start_time

            # 3. Select the accepted tokens. There are two possible cases:
            # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
            # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
            # if do_sample and candidate_logits is not None:
            if candidate_logits is not None:
                start_time = time.time()
                div_threshold = candidate_generator.assistant_model.div_threshold # makeshift solution, should find better way to pass this in
                valid_tokens, n_matches, new_logits, correction_term, divs, acceptance_time, spec_sampling_time = _speculative_backoff_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_logits_unprocessed,
                    candidate_length,
                    new_logits,
                    new_logits_unprocessed if len(div_logits_processor) > 0 else next_token_logits,
                    is_done_candidate,
                    div_threshold,
                    fsd_div_type, 
                    do_sample,
                    logits_processor, 
                    logits_warper,
                    div_logits_processor,
                    cur_len,
                    self.config.eos_token_id, 
                    "classifier" if hasattr(candidate_generator.assistant_model, 'classification_threshold') else "regular"
                )
                
                if hasattr(candidate_generator.assistant_model, "n_matches_list"):
                    candidate_generator.assistant_model.kl_divs.extend(divs.view(-1).tolist())
                    candidate_generator.assistant_model.theoretical_backoffs.append((divs <= div_threshold).int().view(-1).sum())
                    
                    candidate_generator.assistant_model.n_matches_list.append(n_matches + correction_term)
                    # if n_matches == candidate_length:
                    #     candidate_generator.assistant_model.forced_ml_generations += 1
                    candidate_generator.assistant_model.totals_list.append(valid_tokens.shape[-1])
                    candidate_generator.assistant_model.n_discarded_list.append(candidate_input_ids.shape[-1] - n_matches)
                    candidate_generator.assistant_model.candidate_sequences_list.append(candidate_length)

            # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
            # original model logits with the candidate tokens. We can keep the candidate tokens until the first
            # mismatch, or until the max length is reached.
            else:
                if do_sample:
                    probs = new_logits.softmax(dim=-1)
                    selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                else:
                    selected_tokens = new_logits.argmax(dim=-1)
                # print(f"selected_tokens: {selected_tokens}")
                candidate_new_tokens = candidate_input_ids[:, cur_len:]
                n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

                # Ensure we don't generate beyond max_len or an EOS token
                if is_done_candidate and n_matches == candidate_length:
                    n_matches -= 1
                valid_tokens = selected_tokens[:, : n_matches + 1]
            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.
            
            # 4.1. Get the valid continuation, after the matching tokens
            # print(f"input_ids.shape: {input_ids.shape}, valid_tokens.shape: {valid_tokens.shape}")
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            # print(f"output input_ids: {input_ids}")
            if streamer is not None:
                streamer.put(valid_tokens.cpu())
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

            # 5. Update the candidate generation strategy if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
                if output_logits:
                    raw_logits += (next_token_logits,)

                if "past_key_values" not in model_kwargs or start_from_empty_dynamic_cache:
                    added_len = new_cur_len
                    # set it to false for other iterations
                    start_from_empty_dynamic_cache = False
                else:
                    added_len = n_matches + 1
                
                if output_attentions:
                    if self.config.is_encoder_decoder:
                        cross_attentions = _split_model_outputs(
                            cross_attentions, outputs.cross_attentions, cur_len, added_len
                        )
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.decoder_attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                    else:
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                if output_hidden_states:
                    if self.config.is_encoder_decoder:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                        )
                    else:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                        )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=n_matches + 1,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if (
            hasattr(candidate_generator, "assistant_model")
            and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
                candidate_generator.num_assistant_tokens
            )
            
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    initial_start_time = time.time()
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    correction_term = 0
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    acceptance_time = time.time() - initial_start_time
    start_time = time.time()
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1
    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        correction_term = 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t
    print(f"SD: candidate_length: {candidate_length}, n_matches: {n_matches}")# , cur_len: {cur_len}")
    
    spec_sampling_time = time.time() - start_time
    return valid_tokens, n_matches, correction_term, is_accepted, acceptance_time, spec_sampling_time

def _speculative_backoff_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_logits_unprocessed,
    candidate_length,
    new_logits, 
    new_logits_unprocessed,# NOTE: these are unprocessed, unwarped logits
    is_done_candidate,
    div_threshold,
    div_type,
    do_sample, # this is also passed in new
    logits_processor: LogitsProcessorList, # these two must be passed in because we want to work with the logits before they are processed and warped
    logits_warper: Optional[LogitsProcessorList], # these two must be passed in because we want to work with the logits before they are processed and warped
    div_logits_processor: Optional[LogitsProcessorList],
    cur_len,
    eos_token_id,
    candidate_generator_type='classifier',
):
    
    initial_start_time = time.time()
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    correction_term = 0

    if div_type != 'sd':
        
        if div_type == 'random':
            # generate a tensor of shape candidate_logits[:, :, 0].shape with random values between 0 and 1
            divs = torch.rand(candidate_logits[:, :, 0].shape)
        
        if div_type == 'kl_div_processed' or div_type == 'js_div_processed' or div_type == 'tv_div_processed':
            epsilon = 1e-10
            q = candidate_logits.softmax(dim=-1)
            p = new_logits[:, :candidate_length, :].softmax(dim=-1) # need to be cropped because M_L logits include logits for ungenerated position
            
            q_nonzero = (p > 0).int()
            p_nonzero = (q > 0).int()
            both_nonzero = (q_nonzero & p_nonzero).int()
            
            q = q + epsilon
            p = p + epsilon
            
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            
            
        else:
            q = candidate_logits_unprocessed.softmax(dim=-1)
            p = new_logits_unprocessed[:, :candidate_length, :].softmax(dim=-1) # need to be cropped because M_L logits include logits for ungenerated position
            
            if len(div_logits_processor) > 0:
                epsilon = 1e-10
                q = q + epsilon
                p = p + epsilon
                
                p = p / p.sum(dim=-1, keepdim=True)
                q = q / q.sum(dim=-1, keepdim=True)
            
        if div_type == 'kl_div' or div_type == 'kl_div_processed':
            divs = torch.nn.functional.kl_div(torch.log(p), q, reduction='none').sum(dim=-1) # shape = [bs, seq_len]
        elif div_type == 'kl_div_reversed' or div_type == 'kl_div_reversed_processed':
            divs = torch.nn.functional.kl_div(torch.log(q), p, reduction='none').sum(dim=-1) # shape = [bs, seq_len]            
        elif div_type == 'js_div' or div_type == 'js_div_processed':
            m = 0.5 * (p + q)  # Midpoint distribution
            divs = (0.5 * torch.nn.functional.kl_div(torch.log(p), m, reduction='none') + 0.5 * torch.nn.functional.kl_div(torch.log(q), m, reduction='none')).sum(dim=-1)
        elif div_type == 'tv_div' or div_type == 'tv_div_processed':
            divs = 0.5 * torch.abs(p - q).sum(dim=-1)
        
        elif div_type == 'top_p_kl_div' or div_type == 'top_p_js_div' or div_type == 'top_p_tv_div':
            p_sorted, p_sorted_indexes = torch.sort(p, descending=True)
            q_sorted = q[p_sorted_indexes]
            
            cum_p = torch.cumsum(p_sorted, dim=-1)
            
            # Identify the top-p (nucleus) indices
            top_p_mask = cum_p <= top_val
            top_p_mask[torch.argmax(cum_p > top_val)] = True  # Include the first value exceeding p
            top_p = p_sorted[top_p_mask]
            top_q = q_sorted[top_p_mask]

            # Normalize the nucleus probabilities
            top_p = top_p / top_p.sum()
            top_q = top_q / top_q.sum()
            
            if div_type == 'top_p_kl_div':
                divs = torch.nn.functional.kl_div(torch.log(top_p), top_q, reduction='none').sum(dim=-1)
            
            if div_type == 'top_p_js_div':
                m = 0.5 * (top_p + top_q)  # Midpoint distribution
                divs = (0.5 * torch.nn.functional.kl_div(torch.log(top_p), m, reduction='none') + 0.5 * torch.nn.functional.kl_div(torch.log(top_q), m, reduction='none')).sum(dim=-1)
            
            if div_type == 'top_p_tv_div':
                divs = 0.5 * torch.abs(top_p - top_q).sum(dim=-1)
        
        elif div_type == 'top_k_kl_div' or div_type == 'top_k_js_div' or div_type == 'top_k_tv_div':
            top_val = 50
            
            p_top_k, p_top_k_indices = torch.topk(p, top_val, dim=-1)
            q_top_k = torch.gather(q, -1, p_top_k_indices)
            
            top_k_mask = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, p_top_k_indices, True)
            
            non_top_k_mask = ~top_k_mask  # Invert the mask
            p_non_top_k_values = p * non_top_k_mask  # Zero out the top_k values
            q_non_top_k_values = q * non_top_k_mask  # Zero out the top_k values

            # Sum over the non-top_k positions
            p_non_top_k_sum = p_non_top_k_values.sum(dim=-1, keepdim=True)
            q_non_top_k_sum = q_non_top_k_values.sum(dim=-1, keepdim=True)
            
            p_top_k = torch.cat((p_top_k, p_non_top_k_sum), dim=-1)
            q_top_k = torch.cat((q_top_k, q_non_top_k_sum), dim=-1)
            
            if div_type == 'top_k_kl_div':
                divs = torch.nn.functional.kl_div(torch.log(p_top_k), q_top_k, reduction='none').sum(dim=-1)
            
            if div_type == 'top_k_js_div':
                m = 0.5 * (p_top_k + q_top_k)  # Midpoint distribution
                divs = (0.5 * torch.nn.functional.kl_div(torch.log(p_top_k), m, reduction='none') + 0.5 * torch.nn.functional.kl_div(torch.log(q_top_k), m, reduction='none')).sum(dim=-1)
            
            if div_type == 'top_k_tv_div':
                divs = 0.5 * torch.abs(p_top_k - q_top_k).sum(dim=-1)
            
            print(f"divs: {divs}")
            
        is_accepted = divs <= div_threshold
        
        
        print(f"divs: {divs.tolist()} threshold: {div_threshold} div_type: {div_type}")
        
    else: # this is case of regular SD 
        q = candidate_logits_unprocessed.softmax(dim=-1) # depends on whether processing candidate_logits or not
        q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
        p = new_logits.softmax(dim=-1)
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
        probability_ratio = p_i / q_i

        r_i = torch.rand_like(probability_ratio)
        divs = r_i
        is_accepted = r_i <= probability_ratio
    
    acceptance_time = time.time() - initial_start_time
    start_time = time.time()
    
    true_divs = divs

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1 - 
    logit_processing_time = time.time() - start_time
    start_time = time.time()
    
    if candidate_length == n_matches and new_candidate_input_ids[0, -1] == eos_token_id and candidate_generator_type != 'regular' and div_type != 'sd':
        is_done_candidate = True
        
    is_done_time = time.time() - start_time
    start_time = time.time()
    # print(f"is_done_time: {is_done_time}")
    if is_done_candidate and n_matches == candidate_length:
        backoff_count = n_matches
        total = candidate_length
        n_matches -= 1
        correction_term = 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
        
    else:
        if div_type != 'sd':
            p_n_plus_1 = new_logits.softmax(dim=-1)[:, n_matches, :]
            p_prime = p_n_plus_1 # this is the distribution at the position we must sample from to replace the first rejection
            next_tokens = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
            if n_matches > 0:
                valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], next_tokens), dim=-1)
            else:
                valid_tokens = next_tokens
        else:
            gamma = candidate_logits.shape[1]
            p_n_plus_1 = p[:, n_matches, :]
            if n_matches < gamma:
                q_n_plus_1 = q[:, n_matches, :]
                p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
                p_prime.div_(p_prime.sum())
            else:
                p_prime = p_n_plus_1
            t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

            if n_matches > 0:
                valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
            else:
                valid_tokens = t
    
    print(f"SBD: candidate_length: {candidate_length}, n_matches: {n_matches}")
    spec_sampling_time = time.time() - start_time
    total_time = time.time() - initial_start_time
    return valid_tokens, n_matches, new_logits, correction_term, true_divs, acceptance_time, spec_sampling_time

class FSDLlamaForCausalLM(FuzzyGenerationMixin, LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metric tracking fields
        self.n_matches_list = []
        self.candidate_sequences_list = []
        self.n_discarded_list = []
        self.totals_list = []
        self.forced_ml_generations = []
        self.theoretical_backoffs = []
        self.kl_divs = []
        
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return LlamaForCausalLM.prepare_inputs_for_generation(self, input_ids, **kwargs)

class FSDGemma2ForCausalLM(FuzzyGenerationMixin, Gemma2ForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metric tracking fields
        self.n_matches_list = []
        self.candidate_sequences_list = []
        self.n_discarded_list = []
        self.totals_list = []
        self.forced_ml_generations = []
        self.theoretical_backoffs = []
        self.kl_divs = []
        
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return Gemma2ForCausalLM.prepare_inputs_for_generation(self, input_ids, **kwargs)
    
    
class FSDQwen2ForCausalLM(FuzzyGenerationMixin, Qwen2ForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metric tracking fields
        self.n_matches_list = []
        self.candidate_sequences_list = []
        self.n_discarded_list = []
        self.totals_list = []
        self.forced_ml_generations = []
        self.theoretical_backoffs = []
        self.kl_divs = []
        
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return Qwen2ForCausalLM.prepare_inputs_for_generation(self, input_ids, **kwargs)


MODEL_MAPPING = {
    "LlamaForCausalLM": FSDLlamaForCausalLM,
    "Gemma2ForCausalLM": FSDGemma2ForCausalLM,
    "Qwen2ForCausalLM": FSDQwen2ForCausalLM,
}


class FSDAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        # Load config
        config = AutoConfig.from_pretrained(model_name_or_path)

        # Ensure architectures field exists
        model_class_name = getattr(config, "architectures", [None])[0]
        if model_class_name is None:
            raise ValueError(f"Could not determine model class from config: {model_name_or_path}")

        # Find the custom FSD model
        fsd_model_class = MODEL_MAPPING.get(model_class_name, None)
        if fsd_model_class is None:
            raise ValueError(f"Unsupported model class {model_class_name}")

        print(f"Loading model: {fsd_model_class.__name__}")  # Debugging

        # Load and return the model
        return fsd_model_class.from_pretrained(model_name_or_path, *args, **kwargs)



