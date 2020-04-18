#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J "Training_procedure"
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 1
#BSUB -W 8:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -R "select[gpu16gb]"
#BSUB -N

echo "Running job"
python3 src/rubiks/runtrain.py --config data/configs/hpc_run.ini
echo "Finished running job"

