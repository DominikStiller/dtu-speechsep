#!/bin/sh
### General options
### queue
#BSUB -q gpua10
### job name
#BSUB -J train_speechsep_sinusoid_noupsampling
### number of cores
#BSUB -n 4
### all cores on same host
#BSUB -R "span[hosts=1]"
### 2 GPUs, exclusive mode does not work with torch's DDP model
#BSUB -gpu "num=2:mode=shared"
### walltime limit
#BSUB -W 8:00
# memory
#BSUB -R "rusage[mem=10GB]"
### notify upon completion
#BSUB -N
### output and error file
#BSUB -o data/lsf_logs/train_%J.out
#BSUB -e data/lsf_logs/train_%J.err

mkdir -p data/lsf_logs

module load python3/3.9.14
module load cuda/11.7.1

nvidia-smi

bin/private/train_sinusoid.sh
