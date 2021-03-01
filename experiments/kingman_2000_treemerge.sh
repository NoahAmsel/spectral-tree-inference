#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=STDRsubtrees
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --error=log/treemerge.%A_%a.err
#SBATCH --output=log/treemerge.%A_%a.out
#SBATCH --mail-user=mamie.wang@yale.edu
# reference script from https://rcc.uchicago.edu/docs/running-jobs/array/index.html

module load miniconda/4.7.10
source /ysm-gpfs/apps/software/miniconda/4.7.10/bin/activate py2

echo $CONDA_DEFAULT_ENV

scriptDir="/gpfs/ysm/project/kleinstein/mw957/repos/spectral-tree-inference/experiments"
lookupFile=${scriptDir}/kingman_2000_treemerge.lst

taskID=${SLURM_ARRAY_TASK_ID}
rowNum=$(($taskID+1))

echo "[$0 $(date +%Y%m%d-%H%M%S)] [array-start] hostname = $(hostname) SLURM_JOBID = ${SLURM_JOBID}; SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}" >&2

path=$(cat $lookupFile | awk -F"\t" -v taskID=$rowNum '(NR == taskID) { print $1 }')

cd $path

echo $PWD

/usr/bin/time -f "%E %M" /gpfs/ysm/project/kleinstein/mw957/conda_envs/py2/bin/python2.7 /gpfs/ysm/project/kleinstein/mw957/repos/treemerge/python/treemerge.py \
-s STDR_tree.txt \
-t subtree_*.txt \
-m HKY_distance.txt \
-x taxa.txt \
-o treemerge-on-STDR-subtree.txt \
-w . \
-p /gpfs/ysm/project/kleinstein/mw957/repos/treemerge/paup4a168_centos64 


echo "[$0 $(date +%Y%m%d-%H%M%S)] [array-end] hostname = $(hostname) SLURM_JOBID = ${SLURM_JOBID}; SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}" >&2
