#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "v3batchSizes_moreWins"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="run7_batchSizes_moreWindows_warmup100_initLR0.01_v2_usef0"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 10000 --window 200 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 10000 --window 200 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 10000 --window 200 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 10000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs10000_multiTraj_UNknownIC" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 10000 --window 300 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_win200" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 10000 --window 400 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_win300" &
srun --ntasks=1 python3 test_run_v8_hpc.py --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 10000 --window 500 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_win400"
wait
