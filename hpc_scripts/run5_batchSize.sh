#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=9   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="run5_batchSizeSweep"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 0 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_knownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 0 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_multiTraj_knownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 0 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_knownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_knownIC_gpu" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_multiTraj_knownIC_gpu" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_knownIC_gpu" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 1 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_lam1_gpu" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 1 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_multiTraj_lam1_gpu" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 1 --multi_traj 1 --lr 0.1 --epochs 100000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_lam1_gpu"
wait