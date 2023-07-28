#!/bin/bash
#SBATCH -J serialfarm
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --array=0-7
#SBATCH --partition=serial_std
#SBATCH --mem=15G
#SBATCH --mail-user=mail@address.com
#SBATCH --mail-type=end
#SBATCH --export=NONE
#SBATCH --time=02:00:00

export OMP_NUM_THREADS=56

source activate pysamosa
cd ~/repos/pysamosa

python -m pysamosa.main_s6
#python -c "import pprint; import os; pprint.pprint({k: v for k,v in dict(os.environ).items() if 'SLURM' in k})"
#python -c "import os; print(os.cpu_count())"
