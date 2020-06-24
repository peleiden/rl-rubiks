#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J "Training_procedure"
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 1
#BSUB -W 16:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -R "select[gpu32gb]"
#BSUB -N

echo "Running job"
python runtrain.py --config configs/train_ex.ini --location ../rubiks-models/main
python runeval.py --config configs/eval_ex.ini --location ../rubiks-models/main
