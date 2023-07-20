from transformers import AutoModel
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from peft import PeftModel
import torch
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