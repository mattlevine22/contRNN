#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=8   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="run3_smallerModels_known_ICs"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 0 --do_normalization 1 --batch_size 10 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs10" &
srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100" &

srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 1 --do_normalization 1 --batch_size 10 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs10_hybrid" &
srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 1 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_hybrid" &

srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 0 --do_normalization 0 --batch_size 10 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs10_notNormalized" &
srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 0 --do_normalization 0 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_notNormalized" &

srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 1 --do_normalization 0 --batch_size 10 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs10_hybrid_notNormalized" &
srun --ntasks=1 python3 test_run_v6_hpc.py --use_f0 1 --do_normalization 0 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_hybrid_notNormalized" &

wait
