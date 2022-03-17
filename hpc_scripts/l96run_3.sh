#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "l96run_3"   # job name
#SBATCH --output=slurm/S-%x.%j.out
#SBATCH --error=slurm/S-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/tests_v3"
echo "Sending results to $round_dir"
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 0 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 10000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs10000_multiTRAJ_Lay2_width50_relu_dY72" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 1 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 10000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs10000_multiTRAJ_Lay2_width50_relu_dY72_adjoint" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 0 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 20000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs20000_multiTRAJ_Lay2_width50_relu_dY72" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 1 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 20000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs20000_multiTRAJ_Lay2_width50_relu_dY72_adjoint" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 0 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 10000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs10000_multiTRAJ_Lay2_width50_relu_dY72_lr0.01" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 1 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 10000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs10000_multiTRAJ_Lay2_width50_relu_dY72_lr0.01_adjoint" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 0 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 20000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs20000_multiTRAJ_Lay2_width50_relu_dY72_lr0.01" &
srun -l --ntasks=1 python3 test_run_v9_hpc.py --eval_time_limit 600 --adjoint 1 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 20000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation relu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs20000_multiTRAJ_Lay2_width50_relu_dY72_lr0.01_adjoint"
wait
