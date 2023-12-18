#!/bin/bash

#SBATCH -J synth_nono               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -e job.%j.err
#SBATCH -n 1                  # Total number of nodes requested
#SBATCH --partition=milan,rome,cascade               # Partition name
#SBATCH -t 12:00:00           # Run time (hh:mm:ss) 



# module load /usr/local/bin/python3.8
cd ~/synthesis

source ~/myenv/bin/activate
python -u run_synth_problems.py synth_nono
deactivate


