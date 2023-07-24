from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import transformers
from transformers.generation import LogitNormalization

from nltk.util import ngrams
from collections import Counter
import json
import random

import torch.nn.functional as F
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import yaml
import os
import openai

from utils import format_output
from steer import STEER


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
            self.max_length = 75
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
    
class EvaluationPipeline(PipelineBase):

    def __init__(self, task: str, method: str, model: str):
        super().__init__(task, method, model)
        config = yaml.safe_load(open("config.yaml", "r"))
        self.synthetic_dataset, self.real_dataset = self.load_datasets()
        self.deployment_name = config['openai_deployment_embeddings']
        openai.api_key = config['openai_api_key']
        openai.api_base = config['openai_api_base']
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'

    def load_datasets(self):
        if self.task == "commonsense":
            with open("../results/commonsense-falcon-7b-no_steer.txt") as file:
                synthetic_dataset = np.array(["Which "+line.strip() for line in file.read().split("Which")[:-1]])
        else:
            with open(self.synthetic_dataset_path, "r") as file:
                synthetic_dataset = np.array([line.strip() for line in file])
        with open(self.real_dataset_path, "r") as file:
            real_dataset = np.array([example['text'].split(f"{self.split}: ")[-1] for example in json.load(file)])
        return synthetic_dataset, real_dataset
    
    def set_embeddings(self, local_disk = True):
        if local_disk:
            self.synthetic_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-synthetic.npy")
            self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-real.npy")
        else:
            self.synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
            self.real_embeddings = self.embed_dataset(self.real_dataset)
            # Save to ../results/embeddings with the names being the task, model and synthetic/real
            np.save(f"../results/embeddings/{self.task}-{self.model}-synthetic.npy", self.synthetic_embeddings)
            np.save(f"../results/embeddings/{self.task}-{self.model}-real.npy", self.real_embeddings)

    def embed_example(self, example: str):
        response = openai.Embedding.create(
            engine=self.deployment_name,
            input=example,
        )
        return response['data'][0]['embedding']
    
    def embed_dataset(self, dataset) -> np.array:
        embeddings = []
        for example in tqdm(dataset):
            embeddings.append(self.embed_example(example))
        return np.array(embeddings)
    
    ### TRADITIONAL METRICS ###

    def normalised_ngrams(self, tokenizer, n) -> float:
        """
        Calculate the normalised count of unique n-grams in a text.
        """

        # Concatenate the text from the synthetic dataset
        synthetic_text = " ".join(self.synthetic_dataset)
        real_text = " ".join(self.real_dataset)[:len(synthetic_text)]

        n_grams_list = []

        for text in [synthetic_text, real_text]:

            # Tokenize the text
            tokens = tokenizer.tokenize(text)
            
            # Generate n-grams from the token list
            generated_ngrams = list(ngrams(tokens, n))
            
            # Count unique n-grams
            unique_ngrams = len(set(generated_ngrams))

            # Append to the list
            n_grams_list.append(unique_ngrams / len(generated_ngrams) if len(generated_ngrams) > 0 else 0)
        
        # Return the normalised count of unique n-grams
        return {'synthetic': n_grams_list[0], 'real': n_grams_list[1]}

    def convex_hull_area(self, umap_dimensions: int = 2) -> float:
        """
        Calculate the area ratio of the convex hulls of the embeddings of the synthetic and real examples.
        NOTE: the number of synthetic examples should be equal to the number of real examples for constant comparison.
        """
        # Combine synthetic and real embeddings for UMAP fitting
        combined_embeddings = np.vstack((self.synthetic_embeddings, self.real_embeddings))

        # Standardize the embeddings
        combined_embeddings = StandardScaler().fit_transform(combined_embeddings)

        # Reduce dimensionality with UMAP
        umap_embeddings = umap.UMAP(n_components=umap_dimensions, n_neighbors=min(combined_embeddings.shape[0]-1, 50)).fit_transform(combined_embeddings)

        # Split the UMAP embeddings back into synthetic and real
        num_synthetic = len(self.synthetic_embeddings)
        synthetic_umap_embeddings = umap_embeddings[:num_synthetic]
        real_umap_embeddings = umap_embeddings[num_synthetic:]

        # Compute convex hulls
        synthetic_hull = ConvexHull(synthetic_umap_embeddings)
        real_hull = ConvexHull(real_umap_embeddings)

        # Compute and return the ratio of the areas (or volumes)
        if len(synthetic_umap_embeddings[0]) == 2:
            return synthetic_hull.area / real_hull.area
        elif len(synthetic_umap_embeddings[0]) > 2:
            return synthetic_hull.volume / real_hull.volume
        else:
            raise ValueError("Points must have at least two dimensions.")


    def kl_divergence(self):
        synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
        real_embeddings = self.embed_dataset(self.real_dataset)
        pass

    def cosine_similarity(self, centroid: bool = False) -> float:
        """
        Calculate the average or centroid cosine similarity between the synthetic and real datasets.
        """
        synthetic_embeddings = self.synthetic_embeddings
        real_embeddings = self.real_embeddings
        if centroid:
            similarities = cosine_similarity(np.mean(synthetic_embeddings, axis=0).reshape(1, -1), 
                                             np.mean(real_embeddings, axis=0).reshape(1, -1))
        else: similarities = cosine_similarity(synthetic_embeddings, real_embeddings)
        return np.mean(similarities)
    

    
    ### NOVEL METRICS ###
    def authenticity_auroc(self):
        synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
        real_embeddings = self.embed_dataset(self.real_dataset)

        # Instantiate XGBoost model and prepare data
        xgb_model = xgb.XGBClassifier()
        X = np.concatenate((synthetic_embeddings, real_embeddings))
        y = np.concatenate((np.zeros(len(synthetic_embeddings)), np.ones(len(real_embeddings))))

        # Split into train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the model
        xgb_model.fit(X_train, y_train)

        # Get predictions
        y_pred = xgb_model.predict(X_test)

        # Calculate AUROC
        return roc_auc_score(y_test, y_pred)
    
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
        
        sequences = fine_tuned_model.generate(
            input_ids=prompt_tensor['input_ids'],
            attention_mask=prompt_tensor['attention_mask'],
            max_new_tokens=80,
            eos_token_id=42,
            pad_token_id = tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([
                STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                      uncond=neg_prompt, model=fine_tuned_model),
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
        if batch_size >= 0:
            for _ in tqdm(range(num_examples)):
                example = self.generate_response(base_model, fine_tuned_model, tokenizer)
                dataset.append(self.format_example(example, batch=False))

        else:
            texts = [self.prompt]*10
            encoding = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                generated_ids = fine_tuned_model.generate(
                                    input_ids=encoding['input_ids'],
                                    attention_mask=encoding['attention_mask'],
                                    max_new_tokens=80,
                                    eos_token_id=42,
                                    pad_token_id = tokenizer.eos_token_id,
                                    logits_processor=LogitsProcessorList([
                                        STEER(gamma=self.gamma, eta=self.eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                                            uncond=neg_prompt, model=fine_tuned_model),
                                    ]),
                                    do_sample=True,
                                    top_k=10,
                                    num_return_sequences=1,
                                )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            dataset.append(generated_texts)

        if save_to_disk:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            model_name = self.parent_model_path.split("/")[-1]
            file_path = os.path.join(self.output_dir, f"{self.task}-{model_name}-{self.method}.txt")
            with open(file_path, "a") as file:
                for example in dataset:
                    file.write(example + "\n")
        
        return dataset
    
def steer_generation(gamma: float = 0.2, eta = 0.2):
    fine_tuned_model_path = "/g/data/y89/cn1951/falcon-7b-hypotheses-tiny"
    base_model_path = "/g/data/y89/cn1951/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, 
                                                      device_map="auto", trust_remote_code=True, local_files_only=True)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, torch_dtype=torch.bfloat16, 
                                                            device_map="auto", trust_remote_code=True, local_files_only=True)
    
    prompt = tokenizer("### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:", return_tensors='pt')
    
    # You need to define a negative prompt.
    neg_prompt = tokenizer("What is the origin of the core in the Antlia 2 dwarf galaxy and can it be explained by aggressive feedback or alternative theories to cold dark matter?", return_tensors='pt')['input_ids']

    outputs = fine_tuned_model.generate(
        input_ids=prompt['input_ids'].to('cuda'),
        # temperature=0.01,
        attention_mask=prompt['attention_mask'].to('cuda'),
        max_new_tokens=80,
        eos_token_id=42, pad_token_id = tokenizer.eos_token_id,
        logits_processor=LogitsProcessorList([
            STEER(gamma=gamma, eta=eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                  uncond=neg_prompt.to('cuda'), model=fine_tuned_model),
        ]),
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
    )

    print(tokenizer.decode(outputs[0]))


### QUICK RUNS ###

def create_dataset(num_examples: int, save_to_disk: bool = True, batch_size: int = 16):
    # Create the pipeline
    pipeline = InferencePipeline(task="commonsense", method="no_steer", model="falcon-7b")
    # Generate the synthetic dataset
    pipeline.generate_synthetic_dataset(num_examples=num_examples, save_to_disk=save_to_disk, batch_size=batch_size)

def evaluate_model_dataset(task: str, method: str, model: str, local_disk: bool = True):
    pipeline = EvaluationPipeline(task, method, model)
    # Clear the terminal
    os.system("clear")
    pipeline.set_embeddings(local_disk=local_disk)
    print(pipeline.cosine_similarity())
    print(pipeline.convex_hull_area())
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    print(pipeline.normalised_ngrams(tokenizer, 1))

if __name__ == "__main__":
    steer_pipe = STEERPipeline(task="hypotheses", method="steer", model="falcon-7b")
    dataset = steer_pipe.generate_synthetic_dataset(num_examples=90, save_to_disk=True)
    print(dataset)