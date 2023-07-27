from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import transformers
from transformers.generation import LogitNormalization

from nltk.util import ngrams
from collections import Counter
import json
import random
import time

import torch.nn.functional as F
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import umap
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
import os
import openai

from utils import format_output
from steer import STEER
from inference import PipelineBase


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
            self.synthetic_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-synthetic.npy")
            self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-real.npy")
        else:
            self.synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
            self.real_embeddings = self.embed_dataset(self.real_dataset)
            # Save to ../results/embeddings with the names being the task, model and synthetic/real
            np.save(f"../results/embeddings/{self.task}-{self.model}-{self.method}-synthetic.npy", self.synthetic_embeddings)
            np.save(f"../results/embeddings/{self.task}-{self.model}-{self.method}-real.npy", self.real_embeddings)

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

    def authenticity_auroc(self):
        # Create labels
        real_labels = np.ones(len(self.real_embeddings))
        synthetic_labels = np.zeros(len(self.synthetic_embeddings))

        # Combine the data
        data = np.vstack((self.real_embeddings, self.synthetic_embeddings))
        labels = np.concatenate((real_labels, synthetic_labels))

        # Create the k-fold cross-validation object
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create lists to store the results
        roc_auc_scores = []
        accuracy_scores = []

        # Iterate over the folds
        for train_index, test_index in kf.split(data):
            # Split the data
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Train the model
            model = xgb.XGBClassifier()
            model.fit(X_train, y_train)

            # Evaluate the model with accuracy and ROC_AUC
            roc_auc_scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            accuracy_scores.append(accuracy_score(y_test, model.predict(X_test)))

        return np.mean(roc_auc_scores), np.mean(accuracy_scores)

def evaluate_model_dataset(task: str, method: str, model: str, local_disk: bool = True):
    pipeline = EvaluationPipeline(task, method, model)
    # Clear the terminal
    os.system("clear")
    # Retrieve the embeddings
    pipeline.set_embeddings(local_disk=local_disk)
    # Cosine similarity
    cosine_similarity = pipeline.cosine_similarity()
    # Adversarial AUC
    auroc, accuracy = pipeline.authenticity_auroc()
    # Normalised n-grams
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    n_grams = pipeline.normalised_ngrams(tokenizer, 3)
    # Print all the results nicely
    os.system("clear")
    print(f"Model: {model}, Task: {task}, Method: {method}")
    print("-"*80)
    print(f"Cosine similarity: {cosine_similarity:.4f}")
    print(f"Adversarial AUROC: {auroc:.4f}, Adversarial accuracy: {accuracy:.4f}")
    print(f"Normalised n-grams: Synthetic = {n_grams['synthetic']:.4f}, Real = {n_grams['real']:.4f}")
    print("-"*80)