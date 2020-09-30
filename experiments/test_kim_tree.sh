#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=largeTrees
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --error=log/large_trees.%A_%a.err
#SBATCH --output=log/large_trees.%A_%a.out
#SBATCH --mail-user=mamie.wang@yale.edu
# reference script from https://rcc.uchicago.edu/docs/running-jobs/array/index.html

module load Octave
module load miniconda/4.7.10
source /ysm-gpfs/apps/software/miniconda/4.7.10/bin/activate r_env

echo $CONDA_DEFAULT_ENV

scriptDir="/gpfs/ysm/project/kleinstein/mw957/repos/spectral-tree-inference/experiments"
lookupFile=${scriptDir}/test_kim_tree.lst
combineScript=${scriptDir}/test_kim_tree.py

taskID=${SLURM_ARRAY_TASK_ID}
rowNum=$(($taskID+1))

echo "[$0 $(date +%Y%m%d-%H%M%S)] [array-start] hostname = $(hostname) SLURM_JOBID = ${SLURM_JOBID}; SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}" >&2

method=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $1 }')
path=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $2 }')
out=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $3 }')
threshold=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $4 }')
verbose=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $5 }')

echo "/gpfs/ysm/project/kleinstein/mw957/conda_envs/r_env/bin/python3.7 $combineScript $method $path $out --threshold $threshold --verbose $verbose"
/gpfs/ysm/project/kleinstein/mw957/conda_envs/r_env/bin/python3.7 $combineScript $method $path $out --threshold $threshold --verbose $verbose

echo "[$0 $(date +%Y%m%d-%H%M%S)] [array-end] hostname = $(hostname) SLURM_JOBID = ${SLURM_JOBID}; SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}" >&2
