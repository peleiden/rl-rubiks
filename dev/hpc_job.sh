#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J "Training_procedure"
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 1
#BSUB -W 16:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -R "select[gpu32gb]"
#BSUB -N

##
echo "Running job"
# python3 runtrain.py --config configs/main_train.ini --location data/mainextra --rollouts 100
python3 runeval.py --config configs/main_eval.ini --location ../rubiks-models/methodexperiment

#python3 runtrain.py --config configs/method_train.ini --location data/methodexperiment4
#python3 librubiks/solving/hyper_optim.py --location ../rubiks-models/main --agent EGVM

