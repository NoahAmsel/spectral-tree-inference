#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=mut
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --error=log/mut.%A.err
#SBATCH --output=log/mut.%A.out
#SBATCH --mail-user=mamie.wang@yale.edu
# reference script from https://rcc.uchicago.edu/docs/running-jobs/array/index.html
module load Octave
module load miniconda/4.7.10
source activate r_env

python test_mutation_rate.py
