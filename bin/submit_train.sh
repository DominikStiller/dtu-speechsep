#!/bin/sh
### General options
### –- specify queue --
##BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_speechsep
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
##BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o data/lsf_logs/train_%J.out
#BSUB -e data/lsf_logs/train_%J.err
# -- end of LSF options --

mkdir -p data/lsf_logs

module load python3/3.9.14
module load cuda/11.6

#nvidia-smi

. venv/bin/activate

PYTHONPATH=. python speechsep/lightning.py

