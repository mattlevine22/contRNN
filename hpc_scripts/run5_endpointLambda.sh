#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=60:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=10   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="run5_endpointLambda_minlr0.0001"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_knownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam0" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 0.01 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam0.01" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 0.1 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam0.1" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 1 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam1" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 2 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam2" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 5 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam5" &
srun --ntasks=1 python3 test_run_v8_hpc.py --min_lr 0.0001 --lambda_endpoints 10 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam10"
wait
