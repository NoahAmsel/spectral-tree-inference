#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=catepillar
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --error=log/HKYML.%A.err
#SBATCH --output=log/HKYML.%A.out
#SBATCH --mail-user=mamie.wang@yale.edu
# reference script from https://rcc.uchicago.edu/docs/running-jobs/array/index.html
module load Octave
module load miniconda/4.7.10
source activate r_env

python test_catepillar_tree.py
