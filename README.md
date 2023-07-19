# diverse-llm
Improving the diversity of large language models.

## Running on Gadi
The following is a recipe for fine-tuning a model on an interactive compute node on Gadi.

```
qsub -P dg97 -I -lwalltime=06:00:00,ncpus=56,ngpus=4,mem=380GB,jobfs=200GB,wd -qgpursaa
```
after which it will say something like 

```
qsub: waiting for job 90747602.gadi-pbs to start
qsub: job 90747602.gadi-pbs ready
```
