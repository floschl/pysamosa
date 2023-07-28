#!/bin/bash
#SBATCH -J coralfarm
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --array=0-3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=mail@address.com
#SBATCH --export=NONE
#SBATCH --time=15:00:00

module load slurm_setup
export OMP_NUM_THREADS=56

source activate pysamosa

cd ~/repos/pysamosa
srun python -m pysamosa.main_s6
#srun python -c "import os; print(os.environ)"
#srun python -c "import pprint; import os; pprint.pprint({k: v for k,v in dict(os.environ).items() if 'SLURM' in k}); print(f'CPUS: {os.cpu_count()}')"
