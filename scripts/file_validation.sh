#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --time=0:10:00
#SBATCH --mail-user=bhuiyr2@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/file_validation.out
module load python/3.11
source env/bin/activate
echo "Checking that file extensions are valid..."
python src/validate.py