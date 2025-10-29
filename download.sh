#!/bin/bash

# set output and error output filenames, %j will be replaced by Slurm with the jobid
#SBATCH -o download%j.out
#SBATCH -e download%j.err

# single node in the "short" partition
#SBATCH -N 1
#SBATCH -p gpu

#SBATCH --mail-user=s.borad@gwu.edu
#SBATCH --mail-type=ALL

# half hour timelimit
#SBATCH -t 10:00:00

module load python3/3.12.9

cd contrails
source .venv/bin/activate

python3 utils/uwisc_downloader.py