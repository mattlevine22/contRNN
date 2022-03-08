#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "run11_LR0.1"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/run11_DAtests_cheatNormalize_LRinit0.1"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 1 --obs_noise_sd 0.01 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.01_forcing_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 1 --obs_noise_sd 0.1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_forcing_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type forcing --cheat_normalization 1 --obs_noise_sd 1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise1_forcing_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 0.01 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.01_3dvar_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 0.1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise1_3dvar_T1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 0.01 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.01_3dvar_T0.1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 0.1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise1_3dvar_T0.1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 1 --obs_noise_sd 1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 210 --warmup 200 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise1_3dvar_T0.1_warmupT2"
wait
