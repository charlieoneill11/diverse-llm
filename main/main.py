from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
import numpy as np
import torch
from transformers import (LogitsProcessor, LogitsProcessorList,
                          MinLengthLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper,
                          TypicalLogitsWarper)
from transformers.generation import LogitNormalization
import torch.nn.functional as F

class CFGLogits(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `uncond` branch. Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        uncond (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
        smooth_factor (float):
            The interpolation weight for CFG Rescale. 1 means no rescaling, 0 reduces to the conditional scores without
            CFG. Turn it lower if the output degenerates. Lower values allow for higher guidance scale.
    """

    def __init__(self, guidance_scale, uncond, model, rescale_factor=1.0):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.out = None
        self.rescale_factor = rescale_factor

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        if self.out is None:
            self.out = self.model(self.uncond, use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        out = F.log_softmax(out, dim=-1)
        if self.rescale_factor == 1:
            return out
        return self.rescale_factor * out + (1 - self.rescale_factor) * scores
    
class ContrastiveDecoding(LogitsProcessor):
    r"""Logits processor for Contrastive Decoding (CD). The processor computes a reweighting of the logits based on the difference between the fine-tuned and the base model logits, parameterized by the `gamma` value. 

    Args:
        gamma (float):
            The value of gamma for contrastive decoding. A larger gamma downweights token probabilities from the base model more and hence places more emphasis on the domain distribution.
        base_model:
            The base (untrained) model used for contrastive decoding. The base model is only trained on a generic language modeling task.
        fine_tuned_model:
            The fine-tuned model used for contrastive decoding. The fine-tuned model is aware of the nuances and specifics of the target domain due to its fine-tuning.
    """

    def __init__(self, gamma, base_model, fine_tuned_model):
        self.gamma = gamma
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        
        base_out = self.base_model(input_ids, use_cache=True)
        base_logits = F.log_softmax(base_out.logits[0][-1:], dim=-1)

        ft_out = self.fine_tuned_model(input_ids, use_cache=True)
        ft_logits = F.log_softmax(ft_out.logits[0][-1:], dim=-1)

        out = ft_logits - self.gamma * base_logits
        return out


