from transformers import AutoTokenizer
from nltk.util import ngrams
import mauve
import json
import xgboost as xgb
from collections import defaultdict
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
            print(real_dataset[-1])
            # Get random order of indices to shuffle the real dataset array
            indices = np.arange(len(real_dataset))
            np.random.shuffle(indices)
            real_dataset = real_dataset[indices]
        self.synthetic_dataset = synthetic_dataset
        self.real_dataset = real_dataset[:len(synthetic_dataset)]
    
    def set_embeddings(self, local_disk = True, dataset_tuple = None):
        if local_disk:
            self.synthetic_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-synthetic.npy")
            self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-real.npy")
        elif dataset_tuple is not None:
            self.synthetic_embeddings = self.embed_dataset(dataset_tuple[0])
            self.real_embeddings = self.embed_dataset(dataset_tuple[1])
        else:
            print(f"Embedding synthetic data ({len(self.synthetic_dataset)} examples from {self.synthetic_dataset_path})...")
            self.synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
            print(f"Embedding real data ({len(self.real_dataset)} examples from {self.real_dataset_path})...")
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
    
    ### DOWNSTREAM EVALUATION ###
    def parse_and_embed_data(self, dataset):
        questions = []
        answers = []
        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        for line in dataset:
            if "Answer: " not in line:
                continue
            question, answer = line.split("Answer: ")
            question = question.strip()
            answer = answer.strip().split('.')[0]
            answer = answer_map[answer]
            questions.append(question)
            answers.append(answer)
        embeddings = self.embed_dataset(questions)
        answers = np.array(answers)
        return embeddings, answers

    def downstream_classifier(self):
        synthetic_data, synthetic_labels = self.parse_and_embed_data(self.synthetic_dataset[:100])
        real_data, real_labels = self.parse_and_embed_data(self.real_dataset[:100])
        real_data_train, real_data_val, real_labels_train, real_labels_val = train_test_split(real_data, real_labels, test_size=0.2, random_state=42)

        xgb_synthetic = xgb.XGBClassifier()
        xgb_synthetic.fit(synthetic_data, synthetic_labels)

        real_val_preds_synthetic = xgb_synthetic.predict(real_data_val)
        real_val_proba_synthetic = xgb_synthetic.predict_proba(real_data_val)

        xgb_real = xgb.XGBClassifier()
        xgb_real.fit(real_data_train, real_labels_train)

        real_val_preds_real = xgb_real.predict(real_data_val)
        real_val_proba_real = xgb_real.predict_proba(real_data_val)

        print("Training accuracy on synthetic data:", accuracy_score(synthetic_labels, xgb_synthetic.predict(synthetic_data)))
        print("Validation accuracy on real data (synthetic model):", accuracy_score(real_labels_val, real_val_preds_synthetic))
        print("Validation AUROC on real data (synthetic model):", roc_auc_score(real_labels_val, real_val_proba_synthetic, multi_class='ovr'))
        print("Training accuracy on real data:", accuracy_score(real_labels_train, xgb_real.predict(real_data_train)))
        print("Validation accuracy on real data (real model):", accuracy_score(real_labels_val, real_val_preds_real))
        print("Validation AUROC on real data (real model):", roc_auc_score(real_labels_val, real_val_proba_real, multi_class='ovr'))

def evaluate_model_dataset(experiment: Experiment, local_disk: bool = True):
    pipeline = EvaluationPipeline(experiment=experiment)
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

def downstream_evaluation(experiment: Experiment):
    assert experiment.task in ["comments", "commonsense"]
    if experiment.task == "comments":
        pass
    elif experiment.task == "commonsense":
        pipeline = EvaluationPipeline(experiment=experiment)
        # Clear the terminal
        os.system("clear")
        pipeline.downstream_classifier()


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

def calculate_metrics_and_save_heatmaps(gamma_values: list, eta_values: list, n_samples_per_run: int):
    # Define the heatmaps and metrics
    metrics = ['auroc', 'n_grams', 'mauve']
    heatmaps = defaultdict(lambda: np.zeros((len(gamma_values), len(eta_values))))
    
    # Create the experiment
    experiment = Experiment(task="hypotheses", method="steer", model="falcon-7b")

    eval_pipe = EvaluationPipeline(experiment=experiment)
    real_dataset = eval_pipe.real_dataset
    sampled_indices = np.random.choice(real_dataset.shape[0], n_samples_per_run, replace=False)
    real_dataset = real_dataset[sampled_indices]
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

    for i, gamma in enumerate(gamma_values):
        for j, eta in enumerate(eta_values):
            # Load synthetic dataset and set embeddings
            synthetic_dataset = load_synthetic_dataset(gamma, eta)
            eval_pipe.set_embeddings(local_disk=False, dataset_tuple=(real_dataset, synthetic_dataset))

            # Compute metrics
            try: 
                heatmaps['auroc'][i, j], _ = eval_pipe.authenticity_auroc()
            except: 
                heatmaps['auroc'][i, j] = 0.0
            heatmaps['n_grams'][i, j] = compute_n_grams(synthetic_dataset, tokenizer)
            heatmaps['mauve'][i, j] = eval_pipe.mauve()

    # Save the heatmaps
    for metric in metrics:
        np.save(f"../paper/{metric}_heatmap.npy", heatmaps[metric])


def load_synthetic_dataset(gamma, eta):
    file_name = f"../tmp/dataset_gamma_{gamma}_eta_{eta}.txt"
    with open(file_name, 'r') as f:
        synthetic_dataset = f.readlines()
    return synthetic_dataset


def compute_n_grams(synthetic_dataset, tokenizer):
    synthetic_text = " ".join(synthetic_dataset)
    tokens = tokenizer.tokenize(synthetic_text)
    generated_ngrams = list(ngrams(tokens, 3))
    unique_ngrams = len(set(generated_ngrams))
    return unique_ngrams / len(generated_ngrams) if len(generated_ngrams) > 0 else 0


def plot_heatmaps(gamma_values: list, eta_values: list):
    metrics = ['auroc', 'n_grams', 'mauve']
    labels = ['AUROC', 'Norm. $n$-grams', 'Mauve']
    cmap = ["plasma", "plasma", "plasma"]
    fig, axs = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, metric in enumerate(metrics):
        heatmap = np.load(f"../paper/{metric}_heatmap.npy")
        c = axs[i].imshow(heatmap, cmap=cmap[i], interpolation="nearest", extent=[gamma_values[0], gamma_values[-1], eta_values[0], eta_values[-1]])
        axs[i].set_title(labels[i], fontsize=20)
        axs[i].set_xlabel("$\gamma$", fontsize=18)
        if i == 0:
            axs[i].set_ylabel("$\eta$", fontsize=18)
        fig.colorbar(c, ax=axs[i])
        # Change the tick label size
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        axs[i].tick_params(axis='both', which='minor', labelsize=12)

    plt.savefig("../paper/figures/heatmaps.pdf", dpi=250, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    experiment = Experiment(task="comments", method="steer", model="falcon-7b")
    #downstream_evaluation(experiment=experiment)
    evaluate_model_dataset(experiment=experiment, local_disk=False)
    
    # lst = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # # calculate_metrics_and_save_heatmaps(gamma_values=lst, eta_values=lst, n_samples_per_run=25)
    # plot_heatmaps(gamma_values=lst, eta_values=lst)