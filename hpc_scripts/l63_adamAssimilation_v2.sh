#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=8   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="adam_assimilation"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 100 --learn_inits_only 1 --lambda_endpoints 0.01 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam0.01_lr0.1" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 100 --learn_inits_only 1 --lambda_endpoints 0.01 --multi_traj 1 --lr 1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam0.01_lr1" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 100 --learn_inits_only 1 --lambda_endpoints 0.01 --multi_traj 1 --lr 10 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam0.01_lr10" &
wait
