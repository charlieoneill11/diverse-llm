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
        if self.gamma == 0.0: return scores
        
        base_out = self.base_model(input_ids, use_cache=True)
        base_logits = F.log_softmax(base_out.logits[0][-1:], dim=-1)

        ft_out = self.fine_tuned_model(input_ids, use_cache=True)
        ft_logits = F.log_softmax(ft_out.logits[0][-1:], dim=-1)

        out = ft_logits - self.gamma * base_logits
        return out
    
class NegativePrompting(LogitsProcessor):
    def __init__(self, uncond, model, eta):
        self.uncond = uncond
        self.model = model
        self.out = None
        self.eta = eta

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.eta == 0.0: return scores

        if self.out is None:
            self.out = self.model(self.uncond, use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
    
        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        return scores - self.eta * unconditional_logits

class STEER(LogitsProcessor):
    def __init__(self, gamma, eta, base_model, fine_tuned_model, uncond):
        self.contrastive_decoding = ContrastiveDecoding(gamma, base_model, fine_tuned_model)
        self.negative_prompting = NegativePrompting(uncond, fine_tuned_model, eta)

    def __call__(self, input_ids, scores):
        contrastive_scores = self.contrastive_decoding(input_ids, scores)
        return self.negative_prompting(input_ids, contrastive_scores) # steer scores