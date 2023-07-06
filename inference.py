# from transformers import AutoTokenizer, AutoModelForCausalLM
# import transformers
# import torch

# model = "tiiuae/falcon-7b"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
# )
# sequences = pipeline(
#    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
logits = outputs.logits
print(logits)

# convert these logits to probabilities
probs = F.softmax(logits, dim=1)
print(probs)