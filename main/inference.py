from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import transformers
from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
from transformers import (LogitsProcessor, LogitsProcessorList,
                          MinLengthLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper,
                          TypicalLogitsWarper)
from transformers.generation import LogitNormalization

from nltk.util import ngrams
from collections import Counter
import json

import torch.nn.functional as F
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import yaml
import os
import openai

from utils import format_output, format_batch
from main import ContrastiveDecoding

class InferencePipeline:

    def __init__(self, local_model_path: str, parent_model_path: str, task: str, method: str = "no_steer", output_dir: str = "../results"):
        self.local_model_path = local_model_path
        self.parent_model_path = parent_model_path
        self.output_dir = output_dir
        self.task = task
        self.method = method
        assert self.method in ["steer", "no_steer", "base"]
        self.prompt = "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:"
        self.max_length = 500 if self.task == "abstracts" else 100

    def create_pipeline(self):
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=torch.bfloat16, 
                                                                device_map="auto", trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.parent_model_path)
        pipeline = transformers.pipeline(
            "text-generation",
            model=fine_tuned_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        return pipeline, tokenizer
    
    def generate_response(self, pipeline, tokenizer):

        sequences = pipeline(
            self.prompt,
            max_length=self.max_length,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        return format_output(sequences[0]['generated_text'])
    
    def format_batch(hypotheses_list):
        formatted_hypotheses = []
        for hypotheses in hypotheses_list:
            for hypothesis in hypotheses:
                # Split the generated text on '### Hypothesis:' and take the second part
                text = hypothesis['generated_text'].split('### Hypothesis:')[1].strip()
                # Remove excess question marks, replace them with just one
                text = text.rstrip('?') + '?'
                formatted_hypotheses.append(text)
        return formatted_hypotheses
    
    def generate_synthetic_dataset(self, num_examples: int = 100, save_to_disk: bool = False, batch_size: int = 8):
        """
        NOTE: batch_size = 32 seems to perform the fastest.
        """
        dataset = []
        pipeline, tokenizer = self.create_pipeline()

        if batch_size == 0:
            # Just do it sequentially
            for _ in tqdm(range(num_examples)):
                dataset.append(self.generate_response(pipeline, tokenizer))

        else:
            gen_dataset = GenerationDataset(num_examples=num_examples, prompt=self.prompt)
            pipeline.tokenizer.pad_token_id = 42 # EOS token is '?'
            for out in tqdm(pipeline(gen_dataset, max_length=self.max_length, 
                                     do_sample=True, top_k=10, num_return_sequences=1,
                                     eos_token_id=42, pad_token_id=tokenizer.eos_token_id,
                                     batch_size=batch_size), total=len(gen_dataset)):
                dataset.append(out)
            dataset = format_batch(dataset)

        if save_to_disk:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            model_name = self.parent_model_path.split("/")[-1]
            file_path = os.path.join(self.output_dir, f"{self.task}-{model_name}-{self.method}.txt")
            with open(file_path, "a") as file:
                for example in dataset:
                    file.write(example + "\n")
        
        return dataset
    
class GenerationDataset(Dataset):

    def __init__(self, num_examples, prompt):
        self.num_examples = num_examples
        self.prompt = prompt

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        return self.prompt
    
class EvaluationPipeline:

    def __init__(self, synthetic_dataset_path: str, real_dataset_path: str):
        config = yaml.safe_load(open("config.yaml", "r"))
        self.synthetic_dataset_path = synthetic_dataset_path
        self.real_dataset_path = real_dataset_path
        self.model = self.synthetic_dataset_path.split("/")[-1].split("hypotheses-")[-1].split(".txt")[0]
        self.task = self.real_dataset_path.split("/")[-1].split(".")[0] #../data/hypotheses.json should return hypotheses
        self.synthetic_dataset, self.real_dataset = self.load_datasets()
        self.deployment_name = config['openai_deployment_embeddings']
        openai.api_key = config['openai_api_key']
        openai.api_base = config['openai_api_base']
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'

    def load_datasets(self):
        with open(self.synthetic_dataset_path, "r") as file:
            synthetic_dataset = np.array([line.strip() for line in file])
        with open(self.real_dataset_path, "r") as file:
            real_dataset = np.array([example['text'].split("Hypothesis: ")[-1] for example in json.load(file)])
        return synthetic_dataset, real_dataset
    
    def set_embeddings(self, local_disk = True):
        if local_disk:
            self.synthetic_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-synthetic.npy")
            self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-real.npy")
        else:
            self.synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
            self.real_embeddings = self.embed_dataset(self.real_dataset)
            # Save to ../results/embeddings with the names being the task, model and synthetic/real
            np.save(f"../results/embeddings/{self.task}-{self.model}-synthetic.npy", self.synthetic_embeddings)
            np.save(f"../results/embeddings/{self.task}-{self.model}-real.npy", self.real_embeddings)

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
        """
        Calculate the normalised count of unique n-grams in a text.
        """

        # Concatenate the text from the synthetic dataset
        synthetic_text = " ".join(self.synthetic_dataset)
        real_text = " ".join(self.real_dataset)

        n_grams_list = []

        for text in [synthetic_text, real_text]:

            # Tokenize the text
            tokens = tokenizer.tokenize(text)
            
            # Generate n-grams from the token list
            generated_ngrams = list(ngrams(tokens, n))
            
            # Count unique n-grams
            unique_ngrams = len(set(generated_ngrams))

            # Append to the list
            n_grams_list.append(unique_ngrams / len(generated_ngrams) if len(generated_ngrams) > 0 else 0)
        
        # Return the normalised count of unique n-grams
        return {'synthetic': n_grams_list[0], 'real': n_grams_list[1]}

    def convex_hull_area(self, umap_dimensions: int = 2) -> float:
        """
        Calculate the area of the convex hull of the embeddings of the synthetic examples.
        NOTE: the number of synthetic examples should be equal to the number of real examples for constant comparison.
        """
        embeddings = self.synthetic_embeddings
        embeddings = StandardScaler().fit_transform(embeddings)
        umap_embeddings = umap.UMAP(n_components=umap_dimensions, n_neighbors=min(embeddings.shape[0]-1, 50)).fit_transform(embeddings)
        hull = ConvexHull(umap_embeddings)
        if len(umap_embeddings[0]) == 2: return hull.area
        elif len(umap_embeddings[0]) > 2: return hull.volume
        else: raise ValueError("Points must have at least two dimensions.")

    def kl_divergence(self):
        synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
        real_embeddings = self.embed_dataset(self.real_dataset)
        pass

    def cosine_similarity(self, centroid: bool = False) -> float:
        """
        Calculate the average or centroid cosine similarity between the synthetic and real datasets.
        """
        synthetic_embeddings = self.synthetic_embeddings
        real_embeddings = self.real_embeddings
        if centroid:
            similarities = cosine_similarity(np.mean(synthetic_embeddings, axis=0).reshape(1, -1), 
                                             np.mean(real_embeddings, axis=0).reshape(1, -1))
        else: similarities = cosine_similarity(synthetic_embeddings, real_embeddings)
        return np.mean(similarities)
    

    
    ### NOVEL METRICS ###
    def authenticity_auroc(self):
        synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
        real_embeddings = self.embed_dataset(self.real_dataset)

        # Instantiate XGBoost model and prepare data
        xgb_model = xgb.XGBClassifier()
        X = np.concatenate((synthetic_embeddings, real_embeddings))
        y = np.concatenate((np.zeros(len(synthetic_embeddings)), np.ones(len(real_embeddings))))

        # Split into train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the model
        xgb_model.fit(X_train, y_train)

        # Get predictions
        y_pred = xgb_model.predict(X_test)

        # Calculate AUROC
        return roc_auc_score(y_test, y_pred)


    
def contrastive_generation():
    fine_tuned_model_path = "/g/data/y89/cn1951/falcon-7b-abstracts-tiny"
    base_model_path = "/g/data/y89/cn1951/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, trust_remote_code=True)
    
    prompt = tokenizer("### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:", return_tensors='pt')
    
    device='cuda:0'
    base_model.to(device)
    fine_tuned_model.to(device)
    outputs = fine_tuned_model.generate(
        input_ids=prompt['input_ids'].to(device),
        attention_mask=prompt['attention_mask'].to(device),
        max_new_tokens=125,
        logits_processor=LogitsProcessorList([
            ContrastiveDecoding(gamma=1.5, base_model=base_model, fine_tuned_model=fine_tuned_model),
            TemperatureLogitsWarper(0.8),
            TopPLogitsWarper(0.95),
        ]),
        do_sample=True,
    )

    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    task = "hypotheses"
    inf_pipe = InferencePipeline(local_model_path=f"/g/data/y89/cn1951/falcon-7b-{task}-tiny",
                                 parent_model_path="/g/data/y89/cn1951/falcon-7b", task=task, method="no_steer")
    dataset = inf_pipe.generate_synthetic_dataset(num_examples=612, save_to_disk=True, batch_size=64)
    print(dataset)
    # # Define a list of batch sizes you want to test
    # batch_sizes = [8, 16, 32]

    # # Loop over batch sizes and call your function for each batch size
    # for batch_size in batch_sizes:
    #     print(f"Running for batch size {batch_size}, num examples = 64")
    #     dataset = inf_pipe.generate_synthetic_dataset(num_examples=64, save_to_disk=False, batch_size=batch_size)

    # synthetic_dataset_path = "../results/hypotheses-falcon-7b.txt"
    # real_dataset_path = "../data/hypotheses.json"
    # pipeline = EvaluationPipeline(synthetic_dataset_path, real_dataset_path)
    # # Clear the terminal
    # os.system("clear")
    # pipeline.set_embeddings()
    # print(pipeline.cosine_similarity())
    # print(pipeline.convex_hull_area())
    # tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    # print(pipeline.normalised_ngrams(tokenizer, 1))

    

