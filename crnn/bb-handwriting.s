#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=70:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=brownbros_handwriting
#SBATCH --mail-type=END
#SBATCH --mail-user=nmw2@nyu.edu
#SBATCH --output=brownbros_handwriting_%j.out

module purge;
module load anaconda3/5.3.1;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh;
conda activate ./penv;
export PATH=./penv/bin:$PATH;

python3  main.py -r validated -f 2021-04-28_transcriptions_report.csv -e 60