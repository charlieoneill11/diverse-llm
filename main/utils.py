from transformers import AutoModel
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from peft import PeftModel
import torch
import matplotlib.pyplot as plt
import os
from huggingface_hub import create_repo
import numpy as np
import yaml
from huggingface_hub import snapshot_download
import openai

from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams

def normalised_ngrams(synthetic_dataset, real_dataset, tokenizer, n) -> float:
    synthetic_text = " ".join(synthetic_dataset)
    real_text = " ".join(real_dataset)[:len(synthetic_text)]

    n_grams_list = []

    for text in [synthetic_text, real_text]:
        tokens = tokenizer.tokenize(text)
        generated_ngrams = list(ngrams(tokens, n))
        unique_ngrams = len(set(generated_ngrams))
        n_grams_list.append(unique_ngrams / len(generated_ngrams) if len(generated_ngrams) > 0 else 0)
    
    return {'synthetic': n_grams_list[0], 'real': n_grams_list[1]}

def diversity(synthetic_dataset, real_dataset, tokenizer) -> dict:
    diversity_list = []

    for dataset in ["synthetic_dataset", "real_dataset"]:
        norm_values_product = 1.0
        key = "synthetic" if dataset == "synthetic_dataset" else "real"
        for n in range(2, 5):
            norm_n = normalised_ngrams(real_dataset, synthetic_dataset, tokenizer, n)[key] 
            norm_values_product *= (1.0 - norm_n/100)

        diversity_list.append(norm_values_product)
    
    return {'synthetic': diversity_list[0], 'real': diversity_list[1]}


def compute_cosine_similarity(synthetic_embeddings, real_embeddings, centroid: bool = False) -> float:
    if not centroid:
        similarities = cosine_similarity(np.mean(synthetic_embeddings, axis=0).reshape(1, -1), 
                                         np.mean(real_embeddings, axis=0).reshape(1, -1))
    else: 
        similarities = cosine_similarity(synthetic_embeddings, real_embeddings)
    
    return np.mean(similarities)

def save_dataset_to_disk(output_dir, dataset, task, model, method):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{task}-{model}-{method}.txt")
    with open(file_path, "a") as file:
        for example in dataset:
            file.write(example + "\n")

def setup_openai():
    config = yaml.safe_load(open("config.yaml", "r"))
    deployment_name = config['openai_deployment_embeddings']
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_api_base']
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'
    return deployment_name

def download_model_from_hub(repo_id: str, local_dir: str, cache_dir: str):
    snapshot_download(repo_id=repo_id, local_dir=local_dir, cache_dir=cache_dir)


def download_model_from_hub(repo_id: str, local_dir: str, cache_dir: str):
    snapshot_download(repo_id=repo_id, local_dir=local_dir, cache_dir=cache_dir)


def upload_model_to_hub(path_to_model_folder: str, repo_name: str):
    """
    Uploads a model to the HuggingFace Hub.
    Note: you will need to be logged in to HuggingFace to use this.
    """

    # Load the base Falcon-7B model
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Load our fine-tuned model
    model = PeftModel.from_pretrained(model, path_to_model_folder)
    model = model.merge_and_unload()
    model.push_to_hub(repo_name="charlieoneill/"+repo_name)

def format_output(output):
    output = output.replace("\n", " ")  # Replace newline characters with spaces
    output = output.replace("\\n", " ")
    parts = output.split("###")  # Split string at '###'
    
    # Get and clean instruction part
    instruction = parts[1].strip() 
    
    # Get and clean hypothesis part
    hypothesis = parts[2].strip()  
    
    # Format the output
    formatted_output = f"{instruction}\n\n{hypothesis}"
    
    return formatted_output

def format_batch(hypotheses_list):
    formatted_hypotheses = []
    for hypotheses in hypotheses_list:
        for hypothesis in hypotheses:
            # Split the generated text on '### Hypothesis:' and take the second part
            text = hypothesis['generated_text'].split('Hypothesis: ')[1].strip()
            # Remove excess question marks, replace them with just one
            text = text.rstrip('?') + '?'
            formatted_hypotheses.append(text)
    return formatted_hypotheses

def list_repositories(path):
    try:
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if dirs:
            print(f"All repositories in '{path}':")
            for dir in dirs:
                print(dir)
        else:
            print(f"There are no repositories in '{path}'.")
    except FileNotFoundError:
        print(f"The path '{path}' does not exist.")
    except NotADirectoryError:
        print(f"The path '{path}' is not a directory.")
    except PermissionError:
        print(f"Permission denied for path '{path}'.")

def list_files(path):
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if files:
            print(f"All files in '{path}':")
            for file in files:
                print(file)
        else:
            print(f"There are no files in '{path}'.")
    except FileNotFoundError:
        print(f"The path '{path}' does not exist.")
    except NotADirectoryError:
        print(f"The path '{path}' is not a directory.")
    except PermissionError:
        print(f"Permission denied for path '{path}'.")

def plot_umap(embedding, hull=None):
    real_embeddings = embedding[embedding[:, 2] == 0]
    synthetic_embeddings = embedding[embedding[:, 2] == 1]

    plt.scatter(real_embeddings[:, 0], real_embeddings[:, 1], c="blue", alpha=0.5)
    plt.scatter(synthetic_embeddings[:, 0], synthetic_embeddings[:, 1], c="red", alpha=0.5)
    if hull is not None:
        hull_vertices = hull.vertices
        plt.plot(embedding[hull_vertices, 0], embedding[hull_vertices, 1], c="green", lw=2)

    plt.xlabel("TSNE dimension 1")
    plt.ylabel("TSNE dimension 2")
    plt.show()

def upload():
    # from huggingface_hub import login

    # login("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")

    # Load the base Falcon-7B model
    print("Loading base model.")
    model = AutoModelForCausalLM.from_pretrained(
        "/g/data/y89/cn1951/falcon-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Load our fine-tuned model
    model = PeftModel.from_pretrained(model, f"../models/commonsense/checkpoint-1000")
    model = model.merge_and_unload()
    model.save_pretrained(save_directory="/g/data/y89/cn1951/falcon-7b-commonsense-tiny")

def download():
    directory = "/g/data/y89/cn1951"
    snapshot_download(repo_id="meta-llama/Llama-2-13b-hf", local_dir=directory+"/llama-13b", cache_dir=directory)

if __name__ == "__main__":
    upload()