#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "l96run_1"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/run1"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 0 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT3_singleTRAJ_hidden50" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT3_multiTRAJ_hidden50" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 0 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT1_singleTRAJ_hidden50" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT1_multiTRAJ_hidden50" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 0 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 250 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT3_singleTRAJ_hidden250" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 250 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT3_multiTRAJ_hidden250" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 0 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 250 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT1_singleTRAJ_hidden250" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 250 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT1_multiTRAJ_hidden250"
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 10 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT3_multiTRAJ_hidden50_dY10" &
srun --ntasks=1 python3 test_run_v9_hpc.py --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.1 --dim_x 9 --dim_y 75 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 200 --n_layers 3 --activation relu --output_dir "$round_dir/noise0.1_3dvar_T0.1_warmupT3_multiTRAJ_3layerRelu200"
wait
