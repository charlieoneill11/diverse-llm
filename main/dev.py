from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import transformers
from transformers.generation import LogitNormalization
import json
import random
import time

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import os

from utils import format_output
# from steer import STEER

class PipelineBase:

    def __init__(self, task, method, model):
        self.task = task
        self.method = method
        self.model = model
        self.local_model_path = f"/g/data/y89/cn1951/{self.model}-{self.task}-tiny"
        self.parent_model_path = f"/g/data/y89/cn1951/{self.model}"
        self.synthetic_dataset_path = f"../results/{self.task}-{self.model}-{self.method}.txt"
        self.real_dataset_path = f"../data/{self.task}.json"
        self.output_dir = "../results"
        self._set_attributes()
    
    def _set_attributes(self):
        if self.task == "hypotheses":
            self.prompt = "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:"
            self.split = "Hypothesis"
            self.max_length = 100
        elif self.task == "comments":
            self.prompt = "### Instruction: Generate a non-toxic social media comment.\n ### Comment:"
            self.split = "Comment"
            self.max_length = 50
        elif self.task == "commonsense":
            self.prompt = "### Instruction: Generate a multiple-choice question that relies on common-sense to answer.\n ### Multiple-choice question:"
            self.split = "Multiple-choice question"
            self.max_length = 50
        else:
            pass

class InferencePipeline(PipelineBase):

    def __init__(self, task, method, model):
        super().__init__(task, method, model)

    def create_pipeline(self):
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=torch.bfloat16, 
                                                                device_map="auto", trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.parent_model_path)
        pipeline = transformers.pipeline(
            "text-generation",
            model=fine_tuned_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        return pipeline, tokenizer
    
    def generate_response(self, pipeline, tokenizer):

        sequences = pipeline(
            self.prompt,
            max_length=self.max_length,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        return format_output(sequences[0]['generated_text'])
    
    def format_batch(self, examples_list):
        formatted_examples = []
        for examples in examples_list:
            for example in examples:
                text = self.format_example(example, batch=True)
                formatted_examples.append(text)
        return formatted_examples
    
    def format_example(self, example, batch=False):
        # Split the generated text on '### Hypothesis:' and take the second part
        if batch: text = example['generated_text'].split(f'{self.split}:')[1].strip()
        else: text = example.split(f'{self.split}:')[1].strip()
        if self.task == "hypotheses":
            # Remove excess question marks, replace them with just one
            text = text.rstrip('?') + '?'
        elif self.task == "commonsense":
            # Remove everything after the E. Question ... i.e. remove F. -> onwards if it exists
            text = text.split("F.")[0]
        return text
    
    def generate_synthetic_dataset(self, num_examples: int = 100, save_to_disk: bool = False, batch_size: int = 8):
        """
        NOTE: batch_size = 32 seems to perform the fastest.
        """
        dataset = []
        pipeline, tokenizer = self.create_pipeline()

        if batch_size == 0:
            # Just do it sequentially
            for _ in tqdm(range(num_examples)):
                dataset.append(self.generate_response(pipeline, tokenizer))

        else:
            gen_dataset = GenerationDataset(num_examples=num_examples, prompt=self.prompt)
            if self.task == "hypotheses":
                pipeline.tokenizer.pad_token_id = 42 # EOS token is '?'
                end_token = 42
            else:
                pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
                end_token = tokenizer.eos_token_id
            for out in tqdm(pipeline(gen_dataset, max_length=self.max_length, 
                                     do_sample=True, top_k=10, num_return_sequences=1,
                                     eos_token_id=end_token, pad_token_id=end_token,
                                     batch_size=batch_size), total=len(gen_dataset)):
                dataset.append(out)
            dataset = self.format_batch(dataset)

        if save_to_disk:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            model_name = self.parent_model_path.split("/")[-1]
            file_path = os.path.join(self.output_dir, f"{self.task}-{model_name}-{self.method}.txt")
            with open(file_path, "a") as file:
                for example in dataset:
                    file.write(example + "\n")
        
        return dataset
    
class GenerationDataset(Dataset):

    def __init__(self, num_examples, prompt):
        self.num_examples = num_examples
        self.prompt = prompt

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        return self.prompt
    
class STEERPipeline(InferencePipeline):

    def __init__(self, task, method, model, gamma=0.2, eta=0.2, num_neg_prompts=5):
        super().__init__(task, method, model)
        self.gamma = gamma
        self.eta = eta
        self.num_neg_prompts = num_neg_prompts
        self.base_model_path = self.parent_model_path
        with open(self.real_dataset_path, "r") as file:
            self.running_examples = list([example['text'].split(f"{self.split}: ")[-1] for example in json.load(file)])

    def create_pipeline(self):
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=torch.bfloat16, 
                                                                device_map="auto", trust_remote_code=True, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.bfloat16, 
                                                          device_map="auto", trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.parent_model_path)
        
        return base_model, fine_tuned_model, tokenizer

    def generate_negative_prompt(self, tokenizer):
        neg_examples = random.sample(self.running_examples, self.num_neg_prompts)
        neg_examples = ' '.join(neg_examples)
        return tokenizer(neg_examples, return_tensors='pt')['input_ids'].to('cuda')

    def generate_response(self, base_model, fine_tuned_model, tokenizer):
        neg_prompt = self.generate_negative_prompt(tokenizer)

        prompt_tensor = tokenizer(self.prompt, return_tensors='pt').to('cuda')

        if self.task == "hypotheses": end_token = 42
        else: end_token = tokenizer.eos_token_id
        
        sequences = fine_tuned_model.generate(
            input_ids=prompt_tensor['input_ids'],
            attention_mask=prompt_tensor['attention_mask'],
            max_new_tokens=self.max_length,
            eos_token_id=end_token,
            pad_token_id = tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([
                STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                      uncond=neg_prompt),
            ]),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
        )

        decoded_sequence = format_output(tokenizer.decode(sequences[0]))
        self.running_examples.append(decoded_sequence)

        return decoded_sequence
    
    def generate_synthetic_dataset(self, num_examples: int = 10, save_to_disk: bool = False, batch_size: int = 8):
        """
        NOTE: batch_size = 32 seems to perform the fastest.
        """
        dataset = []
        base_model, fine_tuned_model, tokenizer = self.create_pipeline()

        # Just do it sequentially
        if batch_size == 0:
            start = time.time()
            for _ in tqdm(range(num_examples)):
                example = self.generate_response(base_model, fine_tuned_model, tokenizer)
                dataset.append(self.format_example(example, batch=False))
            end = time.time()
            print(f"Took {end-start:.2f} seconds to generate {num_examples} with sequential inference.")

        else:
    
            start = time.time()
            # pipeline = transformers.pipeline(
            #     "text-generation",
            #     model=fine_tuned_model,
            #     tokenizer=tokenizer,
            #     torch_dtype=torch.bfloat16,
            #     trust_remote_code=True,
            #     device_map="auto",
            # )
            # gen_dataset = GenerationDataset(num_examples=num_examples, prompt=self.prompt)
            # if self.task == "hypotheses":
            #     pipeline.tokenizer.pad_token_id = 42 # EOS token is '?'
            #     end_token = 42
            # else:
            #     pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
            #     end_token = tokenizer.eos_token_id
            # for out in tqdm(pipeline(gen_dataset, max_length=self.max_length, 
            #                          do_sample=True, top_k=10, num_return_sequences=1,
            #                          eos_token_id=end_token, pad_token_id=end_token,
            #                          logits_processor=LogitsProcessorList([
            #                             STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
            #                                   uncond=self.generate_negative_prompt(tokenizer)), 
            #                          ]),
            #                          batch_size=batch_size), total=len(gen_dataset)):
            #     print(out)
            #     dataset.append(out)
            # dataset = self.format_batch(dataset)

            #for _ in tqdm(range(num_examples // batch_size)):
                # neg_prompt = self.generate_negative_prompt(tokenizer)
                # prompt_tensor = tokenizer(self.prompt, return_tensors='pt').to('cuda')
                # if self.task == "hypotheses": end_token = 42
                # else: end_token = tokenizer.eos_token_id
                # sequences = fine_tuned_model.generate(
                #     input_ids=prompt_tensor['input_ids'],
                #     attention_mask=prompt_tensor['attention_mask'],
                #     max_new_tokens=self.max_length,
                #     eos_token_id=end_token,
                #     pad_token_id = tokenizer.eos_token_id,
                #     logits_processor=LogitsProcessorList([
                #         STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                #               uncond=neg_prompt),
                #     ]),
                #     do_sample=True,
                #     top_k=10,
                #     num_return_sequences=batch_size,
                # )
                # generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                # generated_texts = [example.split(f"{self.split}: ")[-1].split("?")[0]+"?" for example in generated_texts]
                #dataset.extend(generated_texts)

                # texts = [self.prompt] * batch_size
                # if self.task == "hypotheses": end_token = 42
                # else: end_token = tokenizer.eos_token_id
                # #tokenizer.pad_token_id = tokenizer.eos_token_id
                # #tokenizer.pad_token = tokenizer.eos_token
                # neg_prompt = self.generate_negative_prompt(tokenizer)
                # encoding = tokenizer(texts, return_tensors='pt').to('cuda')
                # generated_ids = fine_tuned_model.generate(input_ids=encoding['input_ids'],
                #                                           attention_mask=encoding['attention_mask'],
                #                                           max_new_tokens=self.max_length, do_sample=True,
                #                                           top_k=10, num_return_sequences=1,
                #                                           logits_processor=LogitsProcessorList([
                #                                                 STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                #                                                       uncond=neg_prompt),
                #                                           ]),
                #                                           eos_token_id=end_token, pad_token_id=tokenizer.eos_token_id)
                # generated_texts = [tokenizer.decode(s, skip_special_tokens=True)
                # generated_texts = [example.split(f"{self.split}: ")[-1].split("?")[0]+"?" for example in generated_texts]
                # dataset.extend(generated_texts)
            for _ in tqdm(range(num_examples // batch_size)):
                texts = [self.prompt]*batch_size
                encoding = tokenizer(texts, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    sequences = fine_tuned_model.generate(
                                        input_ids=encoding['input_ids'],
                                        attention_mask=encoding['attention_mask'],
                                        max_new_tokens=80,
                                        eos_token_id=42,
                                        pad_token_id = tokenizer.eos_token_id,
                                        logits_processor=LogitsProcessorList([
                                            STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                                                uncond=self.generate_negative_prompt(tokenizer)),
                                        ]),
                                        do_sample=True,
                                        top_k=10,
                                        num_return_sequences=1,
                                    )
                #print(sequences)
                generated_texts = [tokenizer.decode(s) for s in sequences]
                #print(generated_texts)
                #generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                dataset.extend(generated_texts)
                #print(dataset)

            end = time.time()
            print(f"Took {end-start:.2f} seconds to generate {num_examples} examples with batching.")

        if save_to_disk:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            model_name = self.parent_model_path.split("/")[-1]
            file_path = os.path.join(self.output_dir, f"{self.task}-{model_name}-{self.method}.txt")
            with open(file_path, "a") as file:
                for example in dataset:
                    file.write(example + "\n")
        
        return dataset
    
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
        # if self.out is None:
        #     self.out = self.fine_tuned_model(self.uncond, use_cache=True)
        # else:
        #     self.out = self.fine_tuned_model(
        #         input_ids[:, -1:],
        #         use_cache=True,
        #         past_key_values=self.out.past_key_values,
        #     )
        self.out = self.fine_tuned_model(self.uncond, use_cache=False)

        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)

        steer_scores = contrastive_scores - self.eta * unconditional_logits

        return steer_scores


### QUICK RUNS ###

def create_dataset(num_examples: int, save_to_disk: bool = True, batch_size: int = 16):
    # Create the pipeline
    pipeline = InferencePipeline(task="commonsense", method="no_steer", model="falcon-7b")
    # Generate the synthetic dataset
    pipeline.generate_synthetic_dataset(num_examples=num_examples, save_to_disk=save_to_disk, batch_size=batch_size)


if __name__ == "__main__":
    steer_pipe = STEERPipeline(task="hypotheses", method="steer", model="falcon-7b", gamma=0.2, eta=0.2, num_neg_prompts=10)
    dataset = steer_pipe.generate_synthetic_dataset(num_examples=100, batch_size=0, save_to_disk=True)
    print(dataset)

    # evaluate_model_dataset(task="hypotheses", method="steer", model="falcon-7b", local_disk=True)