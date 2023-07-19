# diverse-llm
Improving the diversity of large language models.

## Running on Gadi
The following is a recipe for fine-tuning a model on an interactive compute node on Gadi.

```
qsub -P dg97 -I -lwalltime=06:00:00,ncpus=56,ngpus=4,mem=380GB,jobfs=200GB,storage=scratch/dg97+gdata/dg97+scratch/hh5+gdata/hh5,wd -qgpursaa
```
after which it will say something like 

```
qsub: waiting for job 90747602.gadi-pbs to start
qsub: job 90747602.gadi-pbs ready
```

We now need to load the required packages. I've created a conda environment called `diverse-llm` that has everything you need. To activate it, we need to let the interactive node find our conda binary, and then activate our actual environment:

```
module use /g/data/hh5/public/modules
module load conda/analysis3-22.01
conda activate /scratch/dg97/cn1951/diverse-llm
```

To complete the job, simply type `exit` and run:

```
[cn1951@gadi-gpu-rsaa-0001 main]$ exit
logout

qsub: job 90747602.gadi-pbs completed
```
