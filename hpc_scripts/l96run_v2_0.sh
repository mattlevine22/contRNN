#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "l96_partial"   # job name
#SBATCH --output=slurm/S-%x.%j.out
#SBATCH --error=slurm/S-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/partialObs_eps-1_v0"
echo "Sending results to $round_dir"
srun -l python3 test_run_v9_hpc.py --eval_time_limit 6000 --adjoint 0 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 1 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs1000_multiTRAJ_dY72" &
srun -l python3 test_run_v9_hpc.py --eval_time_limit 6000 --adjoint 1 --ds_name L96Meps-1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 0.01 --dim_x 9 --dim_y 72 --plot_interval 1 --max_grad_norm 1 --gpu 1 --multi_traj 1 --lr 0.01 --epochs 1000 --window 110 --warmup 100 --infer_ic 0 --use_f0 1 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/noise0.01_3dvar_T0.1_warmupT1_bs1000_multiTRAJ_dY72_adjoint"
wait
