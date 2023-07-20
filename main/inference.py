from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import transformers
from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
from transformers import (LogitsProcessor, LogitsProcessorList,
                          MinLengthLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper,
                          TypicalLogitsWarper)
from transformers.generation import LogitNormalization

import torch.nn.functional as F
import numpy as np
import torch
import os

from utils import format_output

def create_pipeline(local_model_path: str = "/g/data/y89/cn1951/falcon-7b-abstracts-tiny", 
                    parent_model_path: str = "/g/data/y89/cn1951/falcon-7b"):
    print("Retrieving model.")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    print("Successfully retrieved model. Creating tokenizer and pipeline.")

    tokenizer = AutoTokenizer.from_pretrained(parent_model_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=fine_tuned_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    print("Successfully created tokenizer and pipeline. Generating hypothesis.")

    return pipeline, tokenizer

def generate_hypothesis():
    print(f"Cuda availability = {torch.cuda.is_available()}")
    pipeline, tokenizer = create_pipeline()

    sequences = pipeline(
    "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:",
        max_length=350,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(format_output(sequences[0]['generated_text']))
    
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
    
def contrastive_generation():
    fine_tuned_model_path = "/g/data/y89/cn1951/falcon-7b-abstracts-tiny"
    base_model_path = "/g/data/y89/cn1951/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, trust_remote_code=True)
    
    prompt = tokenizer("### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:", return_tensors='pt')
    
    device='cuda:0'
    base_model.to(device)
    fine_tuned_model.to(device)
    outputs = fine_tuned_model.generate(
        input_ids=prompt['input_ids'].to(device),
        attention_mask=prompt['attention_mask'].to(device),
        max_new_tokens=125,
        logits_processor=LogitsProcessorList([
            ContrastiveDecoding(gamma=1.5, base_model=base_model, fine_tuned_model=fine_tuned_model),
            TemperatureLogitsWarper(0.8),
            TopPLogitsWarper(0.95),
        ]),
        do_sample=True,
    )

    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    generate_hypothesis()