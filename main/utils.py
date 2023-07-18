from transformers import AutoModel
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

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