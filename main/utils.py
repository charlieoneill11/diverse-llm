from transformers import AutoModel
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

def upload_model_to_hub(path_to_model_folder: str, repo_name: str):
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
  model.push_to_hub(repo_name=repo_name)