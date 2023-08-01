from transformers import AutoTokenizer
from nltk.util import ngrams
import mauve
import json
import xgboost as xgb
import umap
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import re
import openai

from inference import Experiment, PipelineBase, STEERPipeline
from utils import setup_openai, normalised_ngrams, compute_cosine_similarity, diversity
from utils import batch_pairwise_distances, ManifoldEstimator, knn_precision_recall_features

import numpy as np
import torch
from time import time
from sklearn.neighbors import NearestNeighbors

class EvaluationPipeline(PipelineBase):

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.deployment_name = setup_openai()
        self._load_datasets()

    def _load_datasets(self):
        with open(self.synthetic_dataset_path, "r") as file:
            synthetic_dataset = np.array([line.strip() for line in file])
        with open(self.real_dataset_path, "r") as file:
            real_dataset = np.array([example['text'].split(f"{self.split}: ")[-1] for example in json.load(file)])
            # Get random order of indices to shuffle the real dataset array
            indices = np.arange(len(real_dataset))
            np.random.shuffle(indices)
            real_dataset = real_dataset[indices]
        self.synthetic_dataset = synthetic_dataset[:1000]
        self.real_dataset = real_dataset[:1000]
    
    def set_embeddings(self, local_disk = True, dataset_tuple = None):
        if local_disk:
            self.synthetic_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-synthetic.npy")
            self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-real.npy")
        elif dataset_tuple is not None:
            self.synthetic_embeddings = self.embed_dataset(dataset_tuple[0])
            self.real_embeddings = self.embed_dataset(dataset_tuple[1])
        else:
            print(f"Embedding synthetic data ({len(self.synthetic_dataset)} examples)...")
            self.synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
            print(f"Embedding real data ({len(self.real_dataset)} examples)...")
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
        return normalised_ngrams(self.synthetic_dataset, self.real_dataset, tokenizer, n)
    
    def diversity_measure(self, tokenizer):
        return diversity(self.synthetic_dataset, self.real_dataset, tokenizer)

    def mauve(self):
        mauve_results = mauve.compute_mauve(p_features=self.real_embeddings, q_features=self.synthetic_embeddings,
                                            verbose=False)
        return mauve_results.mauve

    def cosine_similarity(self, centroid: bool = True) -> float:
        return compute_cosine_similarity(self.synthetic_embeddings, self.real_embeddings, centroid)

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
    
    def alpha_precision_beta_recall(self):
        # Calculate UMAP
        # Combine synthetic and real embeddings for UMAP fitting
        combined_embeddings = np.vstack((self.synthetic_embeddings, self.real_embeddings))

        # Reduce dimensionality with UMAP
        umap_embeddings = umap.UMAP(n_components=2, n_neighbors=150).fit_transform(combined_embeddings)

        # Split the UMAP embeddings back into synthetic and real
        num_synthetic = len(self.synthetic_embeddings)
        synthetic_umap_embeddings = umap_embeddings[:num_synthetic]
        real_umap_embeddings = umap_embeddings[num_synthetic:]

        alpha_precision, beta_recall = knn_precision_recall_features(real_umap_embeddings, synthetic_umap_embeddings, nhood_sizes=[10])
        
        return alpha_precision[0], beta_recall[0]
    
# def train_downstream(experiment: Experiment):
#     if experiment.task == "comments":




def evaluate_model_dataset(experiment: Experiment, local_disk: bool = True):
    pipeline = EvaluationPipeline(experiment=experiment)
    # Clear the terminal
    # os.system("clear")
    # Retrieve the embeddings
    pipeline.set_embeddings(local_disk=local_disk)
    # Cosine similarity
    cosine_similarity = pipeline.cosine_similarity()
    # Adversarial AUC
    auroc, accuracy = pipeline.authenticity_auroc()
    # Normalised n-grams
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    n_grams = pipeline.normalised_ngrams(tokenizer, 3)
    # Diversity
    diversity = pipeline.diversity_measure(tokenizer)
    # MAUVE
    mauve_score = pipeline.mauve()
    # Alpha-precision and beta-recall
    # alpha_precision, beta_recall = pipeline.alpha_precision_beta_recall()
    # Print all the results nicely
    os.system("clear")
    print(f"Model: {experiment.model}, Task: {experiment.task}, Method: {experiment.method}")
    print("-"*80)
    print(f"Cosine similarity: {cosine_similarity:.4f}")
    print(f"MAUVE: {mauve_score:.4f}")
    print(f"Adversarial AUROC: {auroc:.4f}, Adversarial accuracy: {accuracy:.4f}")
    print(f"Normalised n-grams: Synthetic = {n_grams['synthetic']:.4f}, Real = {n_grams['real']:.4f}")
    print(f"Diversity: Synthetic = {diversity['synthetic']:.4f}, Real = {diversity['real']:.4f}")
    # print(f"Alpha-precision: {alpha_precision:.4f}, Beta-recall: {beta_recall:.4f}")
    print("-"*80)

def generate_heatmap_datasets(gamma_values: list, eta_values: list, n_samples_per_run: int = 10):
    # Create the experiment
    experiment = Experiment(task="hypotheses", method="steer", model="falcon-7b")

    # Create evaluation pipeline to handle embeddings
    eval_pipe = EvaluationPipeline(experiment=experiment)

    # Get real dataset by randomly sampling n_samples_per_run
    real_dataset = eval_pipe.real_dataset
    sampled_indices = np.random.choice(real_dataset.shape[0], n_samples_per_run, replace=False)
    real_dataset = real_dataset[sampled_indices]
    steer_pipe = STEERPipeline(experiment=experiment, gamma=0.0, eta=0.0, num_neg_prompts=10)

    for gamma in tqdm(gamma_values):
        for eta in tqdm(eta_values):
            synthetic_dataset = steer_pipe.generate_synthetic_dataset(num_examples=n_samples_per_run, 
                                                                      batch_size=0, save_to_disk=False,
                                                                      gamma=gamma, eta=eta)

            # Save the synthetic datasets locally as txt
            if not os.path.exists("../tmp"):
                os.mkdir("../tmp")

            file_path = f"../tmp/dataset_gamma_{gamma}_eta_{eta}.txt"
            with open(file_path, 'a') as f:
                for example in synthetic_dataset:
                    f.write(f'{example}\n')

def calculate_metrics_and_create_heatmaps(gamma_values: list, eta_values: list, n_samples_per_run: int):

    # Create the heatmaps
    cosine_similarity_heatmap = np.zeros((len(gamma_values), len(eta_values)))
    auroc_heatmap = np.zeros((len(gamma_values), len(eta_values)))
    n_grams_heatmap = np.zeros((len(gamma_values), len(eta_values)))

    # Create the experiment
    experiment = Experiment(task="hypotheses", method="steer", model="falcon-7b")

    eval_pipe = EvaluationPipeline(experiment=experiment)
    real_dataset = eval_pipe.real_dataset
    sampled_indices = np.random.choice(real_dataset.shape[0], n_samples_per_run, replace=False)
    real_dataset = real_dataset[sampled_indices]
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

    for i, gamma in enumerate(gamma_values):
        for j, eta in enumerate(eta_values):

            # Load the synthetic dataset from the txt file
            file_name = f"../tmp/dataset_gamma_{gamma}_eta_{eta}.txt"
            with open(file_name, 'r') as f:
                synthetic_dataset = f.readlines()

            # Create evaluation pipeline to handle embeddings
            eval_pipe = EvaluationPipeline(experiment=experiment)
            eval_pipe.set_embeddings(local_disk=False, dataset_tuple=(real_dataset, synthetic_dataset))

            # Normalised n-grams
            print(len(synthetic_dataset))
            synthetic_text = " ".join(synthetic_dataset)
            tokens = tokenizer.tokenize(synthetic_text)
            generated_ngrams = list(ngrams(tokens, 3))
            unique_ngrams = len(set(generated_ngrams))
            n_grams = unique_ngrams / len(generated_ngrams) if len(generated_ngrams) > 0 else 0
            print(n_grams)

            # Cosine similarity
            cosine_similarity = eval_pipe.cosine_similarity()
            # Adversarial AUC
            try: auroc, accuracy = eval_pipe.authenticity_auroc()
            except: auroc = 0.0

            # print(cosine_similarity, auroc, n_grams["synthetic"])

            # Add to heatmaps
            cosine_similarity_heatmap[i, j] = cosine_similarity
            auroc_heatmap[i, j] = auroc
            n_grams_heatmap[i, j] = n_grams

    print(cosine_similarity_heatmap)
    print(auroc_heatmap)
    print(n_grams_heatmap)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Heatmaps for STEER")

    # Cosine similarity
    c1 = axs[0].imshow(cosine_similarity_heatmap, cmap="hot", interpolation="nearest", extent=[gamma_values[0], gamma_values[-1], eta_values[0], eta_values[-1]])
    axs[0].set_title("Cosine similarity")
    axs[0].set_xlabel("Gamma")
    axs[0].set_ylabel("Eta")
    fig.colorbar(c1, ax=axs[0])

    # Adversarial AUROC
    c2 = axs[1].imshow(auroc_heatmap, cmap="hot", interpolation="nearest", extent=[gamma_values[0], gamma_values[-1], eta_values[0], eta_values[-1]])
    axs[1].set_title("Adversarial AUROC")
    axs[1].set_xlabel("Gamma")
    axs[1].set_ylabel("Eta")
    fig.colorbar(c2, ax=axs[1])

    # Normalised n-grams
    c3 = axs[2].imshow(n_grams_heatmap, cmap="hot", interpolation="nearest", extent=[gamma_values[0], gamma_values[-1], eta_values[0], eta_values[-1]])
    axs[2].set_title("Normalised n-grams")
    axs[2].set_xlabel("Gamma")
    axs[2].set_ylabel("Eta")
    fig.colorbar(c3, ax=axs[2])

    # Save the figure
    plt.savefig("../paper/figures/heatmaps.pdf", dpi=250, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    experiment = Experiment(task="commonsense", method="greedy", model="falcon-7b")
    evaluate_model_dataset(experiment=experiment, local_disk=False)
    # gamma_lst = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # eta_lst = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # #generate_heatmap_datasets(gamma_values = gamma_lst, eta_values = eta_lst, n_samples_per_run = 25)
    # calculate_metrics_and_create_heatmaps(gamma_values=gamma_lst, eta_values=eta_lst, n_samples_per_run=25)