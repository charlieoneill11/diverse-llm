from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
import numpy as np
import torch
from transformers import LogitsProcessor
from transformers.generation import LogitNormalization
import torch.nn.functional as F

class STEER(LogitsProcessor):
    def __init__(self, gamma, eta, base_model, fine_tuned_model, uncond):
        self.gamma = gamma
        self.eta = eta
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model
        self.uncond = uncond
        self.out, self.base_out, self.ft_out = None, None, None

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.gamma == 0.0 and self.eta == 0.0:
            return scores

        # Contrastive decoding
        self.base_out = self.base_model(input_ids, use_cache=False)
        base_logits = F.log_softmax(self.base_out.logits[0][-1:], dim=-1)

        self.ft_out = self.fine_tuned_model(input_ids, use_cache=False)
        ft_logits = F.log_softmax(self.ft_out.logits[0][-1:], dim=-1)

        contrastive_scores = (1 + self.gamma) * ft_logits - self.gamma * base_logits

        # Negative prompting
        if self.out is None:
            self.out = self.fine_tuned_model(self.uncond, use_cache=True)
        else:
            self.out = self.fine_tuned_model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )

        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)

        steer_scores = contrastive_scores - self.eta * unconditional_logits

        return steer_scores