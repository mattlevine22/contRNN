#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test3"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="experiments/run10_DAtests"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 1 --obs_noise_sd 0 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_noise0_Cheat_Normalize_forcing_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 0 --obs_noise_sd 0 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 0 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_noise0_noCheat_noNormalize_forcing_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 0 --obs_noise_sd 0 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_noise0_noCheat_Normalize_forcing_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 1 --obs_noise_sd 0 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_noise0_Cheat_Normalize_forcing_T0.1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 0 --obs_noise_sd 0 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 0 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_noise0_noCheat_noNormalize_forcing_T0.1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 0 --obs_noise_sd 0 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --min_lr 0.000001 --lambda_endpoints 0 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs1000_multiTraj_UNknownIC_noise0_noCheat_Normalize_forcing_T0.1"
wait
