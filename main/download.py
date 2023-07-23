import numpy as np
from huggingface_hub import snapshot_download
from datasets import disable_caching

if __name__ == "__main__":
    directory = "/g/data/y89/cn1951"
    snapshot_download(repo_id="EleutherAI/gpt-neo-1.3B", local_dir=directory+"/gpt-neo-1.3B", cache_dir=directory)
