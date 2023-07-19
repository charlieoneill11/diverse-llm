# diverse-llm
Improving the diversity of large language models.

## Running on Gadi
The following is a recipe for fine-tuning a model on an interactive compute node on Gadi.

```
qsub -I -lwalltime=06:00:00,ncpus=56,ngpus=4,mem=380GB,jobfs=200GB,wd -qgpursaa
```
