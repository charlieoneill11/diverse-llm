from transformers import AutoModel
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from peft import PeftModel
import torch
import umap
import matplotlib.pyplot as plt
import os

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

def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=data, s=100)
    plt.title(title, fontsize=18)