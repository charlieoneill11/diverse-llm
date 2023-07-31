from transformers import AutoTokenizer
from nltk.util import ngrams
import mauve
import json
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
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

import numpy as np
import torch
from time import time
from sklearn.neighbors import NearestNeighbors

# Batch pairwise distances function stays mostly the same
# but uses PyTorch functions instead of TensorFlow
def batch_pairwise_distances(U, V):
    """Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = torch.sum(U ** 2, dim=1)
    norm_v = torch.sum(V ** 2, dim=1)

    # norm_u as a column and norm_v as a row vectors.
    norm_u = norm_u.view(-1, 1)
    norm_v = norm_v.view(1, -1)

    # Pairwise squared Euclidean distances.
    D = torch.max(norm_u - 2 * torch.mm(U, V.t()) + norm_v, torch.tensor([0.0]))

    return D

class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, features, nhood_sizes=[3], eps=1e-5):
        """Estimate the manifold of given feature vectors.
        
            Args:
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                eps (float): Small number for numerical stability.
        """
        self.nhood_sizes = nhood_sizes
        self.eps = eps
        self.features = torch.Tensor(features)
        self.nbrs = NearestNeighbors(n_neighbors=max(nhood_sizes)+1, algorithm='ball_tree').fit(features)

    def evaluate(self, eval_features, return_realism=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        distances, indices = self.nbrs.kneighbors(eval_features)
        batch_predictions = np.zeros([num_eval_images, len(self.nhood_sizes)], dtype=np.int32)
        for i, size in enumerate(self.nhood_sizes):
            for j in range(num_eval_images):
                D = torch.Tensor(distances[j, 1:size+1])
                distance_batch = batch_pairwise_distances(torch.Tensor([eval_features[j]]), self.features[indices[j, 1:size+1]])
                samples_in_manifold = distance_batch <= D
                batch_predictions[j, i] = np.any(samples_in_manifold.numpy(), axis=1).astype(np.int32)
        return batch_predictions

def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3]):
    """Calculates k-NN precision and recall for two sets of feature vectors.
    
        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.

        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    state = dict()

    # Initialize ManifoldEstimators.
    ref_manifold = ManifoldEstimator(ref_features, nhood_sizes) 
    eval_manifold = ManifoldEstimator(eval_features, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall...')
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state['precision'], state['recall']

class EvaluationPipeline(PipelineBase):

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.synthetic_dataset, self.real_dataset = self.load_datasets()
        self.deployment_name = setup_openai()

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
    
    def set_embeddings(self, local_disk = True, dataset_tuple = None):
        if local_disk:
            self.synthetic_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-synthetic.npy")
            self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-real.npy")
        elif dataset_tuple is not None:
            self.synthetic_embeddings = self.embed_dataset(dataset_tuple[0])
            self.real_embeddings = self.embed_dataset(dataset_tuple[1])
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
        return normalised_ngrams(self.synthetic_dataset, self.real_dataset, tokenizer, n)
    
    def diversity_measure(self, tokenizer):
        return diversity(self.synthetic_dataset, self.real_dataset, tokenizer)

    def mauve(self):
        mauve_results = mauve.compute_mauve(p_features=self.real_embeddings, q_features=self.synthetic_embeddings,
                                            verbose=False)
        return mauve_results.mauve

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
    diversity = pipeline.diversity()
    # MAUVE
    mauve_score = pipeline.mauve()
    # Alpha-precision and beta-recall
    alpha_precision, beta_recall = pipeline.alpha_precision_beta_recall()
    # Print all the results nicely
    os.system("clear")
    print(f"Model: {experiment.model}, Task: {experiment.task}, Method: {experiment.method}")
    print("-"*80)
    print(f"Cosine similarity: {cosine_similarity:.4f}")
    print(f"MAUVE: {mauve_score:.4f}")
    print(f"Adversarial AUROC: {auroc:.4f}, Adversarial accuracy: {accuracy:.4f}")
    print(f"Normalised n-grams: Synthetic = {n_grams['synthetic']:.4f}, Real = {n_grams['real']:.4f}")
    print(f"Diversity: Synthetic = {diversity['synthetic']:.4f}, Real = {diversity['real']:.4f}")
    print(f"Alpha-precision: {alpha_precision:.4f}, Beta-recall: {beta_recall:.4f}")
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
    experiment = Experiment(task="hypotheses", method="steer", model="falcon-7b")
    evaluate_model_dataset(experiment=experiment, local_disk=True)
    # gamma_lst = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # eta_lst = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # #generate_heatmap_datasets(gamma_values = gamma_lst, eta_values = eta_lst, n_samples_per_run = 25)
    # calculate_metrics_and_create_heatmaps(gamma_values=gamma_lst, eta_values=eta_lst, n_samples_per_run=25)