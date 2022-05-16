#!/bin/bash
#SBATCH --job-name=bb_train
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yw3076@nyu.edu

module load cuda/11.3.1
singularity exec --nv --overlay /scratch/yw3076/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate ds

python3 /scratch/yw3076/bb-handwriting-model/crnn/main.py -r validated_onepass -f 2022-03-28_transcriptions_report.csv -e 200"

