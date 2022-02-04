#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno
srun --ntasks=1 python3 test_run_v4_hpc.py --window 1000 --hpc 1 --known_inits 1 --output_dir l63/test_givenIC_normalizeICs_HYBRID_T10 &
srun --ntasks=1 python3 test_run_v4_hpc.py --window 500 --hpc 1 --known_inits 1 --output_dir l63/test_givenIC_normalizeICs_HYBRID_T5 &
wait
