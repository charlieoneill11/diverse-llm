#!/bin/bash
 
#PBS -l ncpus=24
#PBS -l mem=10GB
#PBS -l jobfs=10GB
#PBS -q normal
#PBS -P 
#PBS -l walltime=00:01:00
#PBS -l storage=gdata/dg97+scratch/dg97
#PBS -l wd
  
module load python3/3.10.4
python3 main.py $PBS_NCPUS > /g/data/dg97/$USER/job_logs/$PBS_JOBID.log