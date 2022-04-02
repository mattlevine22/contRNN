#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "l63rerun"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/rerunL63"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise1_3dvar_warmupT3_T0.1_gelu50_lr0.1" &
srun --ntasks=1 python3 test_run_v9_hpc.py --warmup_type enkf --cheat_normalization 0 --obs_noise_sd 1 --dim_x 1 --dim_y 2 --plot_interval 10 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.1 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/noise1_enkf_warmupT3_T0.1_gelu50_lr0.1"
wait
