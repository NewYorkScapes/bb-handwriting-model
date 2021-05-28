#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=70:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=brownbros_handwriting
#SBATCH --mail-type=END
#SBATCH --mail-user=nmw2@nyu.edu
#SBATCH --output=brownbros_handwriting_%j.out
#SBATCH --array=0-4

module purge;
module load anaconda3/2020.07;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/nmw2/bb-handwriting-model/penv;
export PATH=/scratch/nmw2/bb-handwriting-model/penv/bin:$PATH;

runopts=(validated validated_onepass validated_iam validated_onepass_iam 
onepass_iam)

rtype=${runopts[$SLURM_ARRAY_TASK_ID]}

python3 /scratch/nmw2/bb-handwriting-model/crnn/main.py -r $rtype -f 2021-05-28_transcriptions_report.csv -e 60
