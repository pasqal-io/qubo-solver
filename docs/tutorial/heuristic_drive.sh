#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --job-name=heuristic_drive
#SBATCH --output=heuristic_drive%j.log
source ~.local/share/hatch/env/virtual/qubo-solver/IwRVwNQb/qubo-solver/bin/activate

srun python /home/ynaghmouchi/qubo-solver/docs/tutorial/heuristic_drive.py