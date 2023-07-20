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
from tqdm import tqdm
import numpy as np
import torch
import os

from utils import format_output
from main import ContrastiveDecoding

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

def generate_abstract():
    print(f"Cuda availability = {torch.cuda.is_available()}")
    pipeline, tokenizer = create_pipeline()

    sequences = pipeline(
    "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:",
        max_length=500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    return format_output(sequences[0]['generated_text'])

def generate_hypothesis():
    local_model_path = "/g/data/y89/cn1951/falcon-7b-hypotheses-tiny"
    parent_model_path = "/g/data/y89/cn1951/falcon-7b"

    print(f"Cuda availability = {torch.cuda.is_available()}")
    pipeline, tokenizer = create_pipeline(local_model_path=local_model_path, parent_model_path=parent_model_path)

    sequences = pipeline(
    "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:",
        max_length=100,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    return format_output(sequences[0]['generated_text'])

def generate_synthetic_dataset(num_examples: int = 100, task: str = "abstracts", model: str = "falcon-7b"):
    dataset = []
    for i in tqdm(range(num_examples)):
        dataset.append(generate_abstract())

    output_dir = "../results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{task}-{model}-{num_examples}.txt")
    with open(file_path, "w") as file:
        for example in dataset:
            file.write(example + "\n")
    
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
    hypothesis = generate_hypothesis()
    print(hypothesis)