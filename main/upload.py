from transformers import AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import create_repo
import torch
from huggingface_hub import login

login("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")

# Load the base Falcon-7B model
model = AutoModelForCausalLM.from_pretrained(
    "/g/data/y89/cn1951/falcon-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# Load our fine-tuned model
model = PeftModel.from_pretrained(model, f"../models/checkpoint-1000")
model = model.merge_and_unload()
# model.save_pretrained(save_directory="/g/data/y89/cn1951/falcon-7b-hypotheses-tiny")

create_repo(f"universeTBD/falcon-7b-hypotheses-tiny", private=False)
model.push_to_hub("universeTBD/falcon-7b-hypotheses-tiny")