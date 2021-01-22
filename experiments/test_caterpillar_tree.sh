#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=caterpillar
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=10G
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
lookupFile=${scriptDir}/test_caterpillar_tree.lst
combineScript=${scriptDir}/test_caterpillar_tree.py

taskID=${SLURM_ARRAY_TASK_ID}
rowNum=$(($taskID+1))

echo "[$0 $(date +%Y%m%d-%H%M%S)] [array-start] hostname = $(hostname) SLURM_JOBID = ${SLURM_JOBID}; SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}" >&2

type=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $1 }')
method=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $2 }')
nrun=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $3 }')
size=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $4 }')
path=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $5 }')
threshold=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $6 }')
m=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $7 }')
kappa=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $8 }')
mutation_rate=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $9 }')
verbose=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $10 }')

echo "/gpfs/ysm/project/kleinstein/mw957/conda_envs/r_env/bin/python3.7 $combineScript $type $method $nrun --size $size --path $path --threshold $threshold --m $m --kappa $kappa --mutation_rate $mutation_rate --verbose $verbose"
/gpfs/ysm/project/kleinstein/mw957/conda_envs/r_env/bin/python3.7 $combineScript $type $method $nrun --size $size --path $path --threshold $threshold --m $m --kappa $kappa --mutation_rate $mutation_rate --verbose $verbose

echo "[$0 $(date +%Y%m%d-%H%M%S)] [array-end] hostname = $(hostname) SLURM_JOBID = ${SLURM_JOBID}; SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}" >&2
