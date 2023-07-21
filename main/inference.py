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

from nltk.util import ngrams
from collections import Counter

import torch.nn.functional as F
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import torch
import yaml
import os
import openai

from utils import format_output
from main import ContrastiveDecoding

class InferencePipeline:

    def __init__(self, local_model_path: str, parent_model_path: str, task: str, output_dir: str = "../results"):
        self.local_model_path = local_model_path
        self.parent_model_path = parent_model_path
        self.output_dir = output_dir
        self.task = task
        self.prompt = "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:"
        self.max_length = 500 if self.task == "abstracts" else 150

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
    
    def generate_synthetic_dataset(self, num_examples: int = 100, save_to_disk: bool = False):
        dataset = []
        pipeline, tokenizer = self.create_pipeline()
        for _ in tqdm(range(num_examples)):
            dataset.append(self.generate_response(pipeline, tokenizer))

        if save_to_disk:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            file_path = os.path.join(self.output_dir, f"{self.task}-{num_examples}.txt")
            with open(file_path, "a") as file: # Change "w" to "a" here
                for example in dataset:
                    file.write(example.split("Hypothesis: ")[1] + "\n")
        
        return dataset
    
class EvaluationPipeline:

    def __init__(self, synthetic_dataset: np.array, real_dataset: np.array):
        config = yaml.safe_load(open("../config.yaml", "r"))
        self.synthetic_dataset = synthetic_dataset
        self.real_dataset = real_dataset
        self.deployment_name= config['openai_deployment_embeddings']
        openai.api_key = config['openai_api_key']
        openai.api_base = config['openai_api_base']
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'
        # TODO: tokenizer for normalised n-grams
        # TODO: convex hull calculator for diversity
        # TODO: KL divergence calculator
        # TODO: cosine similarity score with real dataset

    @staticmethod
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

    def normalised_ngrams(self) -> float:
        pass

    def normalized_ngrams(tokenizer, text, n) -> float:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        
        # Generate n-grams from the token list
        generated_ngrams = list(ngrams(tokens, n))
        
        # Count unique n-grams
        unique_ngrams = len(set(generated_ngrams))
        
        # Return the normalized count of unique n-grams
        return unique_ngrams / len(generated_ngrams) if len(generated_ngrams) > 0 else 0

    def convex_hull_area(self) -> float:
        """
        Calculate the area of the convex hull of the embeddings of the generated examples.
        """
        embeddings = self.embed_dataset(self.synthetic_dataset)
        hull = ConvexHull(embeddings)
        if len(embeddings[0]) == 2: return hull.area
        elif len(embeddings[0]) > 2: return hull.volume
        else: raise ValueError("Points must have at least two dimensions.")

    def kl_divergence(self):
        synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
        real_embeddings = self.embed_dataset(self.real_dataset)
        pass

    def cosine_similarity(self, centroid: bool = False) -> float:
        """
        Calculate the average or centroid cosine similarity between the synthetic and real datasets.
        """
        synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
        real_embeddings = self.embed_dataset(self.real_dataset)
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
    task = "hypotheses"
    inf_pipe = InferencePipeline(local_model_path=f"/g/data/y89/cn1951/falcon-7b-{task}-tiny",
                                 parent_model_path="/g/data/y89/cn1951/falcon-7b", task=task)
    dataset = inf_pipe.generate_synthetic_dataset(num_examples=10, save_to_disk=True)
    print(dataset)