#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "l63_mask"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/maskHidden_L63_v0-1"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v9_hpc.py --lambda_mask 0.01 --mask_hidden 1 --infer_K 1 --multi_traj 0 --dim_x 1 --dim_y 2 --backprop_warmup 1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 1  --plot_interval 10 --max_grad_norm 1 --gpu 1  --lr 0.01 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --shuffle 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/lambda0.01_dy2" &
srun --ntasks=1 python3 test_run_v9_hpc.py --lambda_mask 0.01 --mask_hidden 1 --infer_K 1 --multi_traj 0 --dim_x 1 --dim_y 10 --backprop_warmup 1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 1  --plot_interval 10 --max_grad_norm 1 --gpu 1  --lr 0.01 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --shuffle 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/lambda0.01_dy10" &
srun --ntasks=1 python3 test_run_v9_hpc.py --lambda_mask 0.01 --mask_hidden 1 --infer_K 1 --multi_traj 0 --dim_x 1 --dim_y 10 --backprop_warmup 1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 1  --plot_interval 10 --max_grad_norm 1 --gpu 1  --lr 0.01 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --shuffle 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/lambda0.001_dy10" &
srun --ntasks=1 python3 test_run_v9_hpc.py --lambda_mask 0.01 --mask_hidden 1 --infer_K 1 --multi_traj 0 --dim_x 1 --dim_y 10 --backprop_warmup 1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 1  --plot_interval 10 --max_grad_norm 1 --gpu 1  --lr 0.01 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --shuffle 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/lambda0.1_dy10" &
srun --ntasks=1 python3 test_run_v9_hpc.py --lambda_mask 0.01 --mask_hidden 1 --infer_K 1 --multi_traj 0 --dim_x 1 --dim_y 10 --backprop_warmup 1 --warmup_type 3dvar --cheat_normalization 0 --obs_noise_sd 1  --plot_interval 10 --max_grad_norm 1 --gpu 1  --lr 0.01 --epochs 1000 --window 310 --warmup 300 --infer_ic 0 --use_f0 0 --do_normalization 1 --shuffle 1 --batch_size 1000 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/lambda1_dy10"
wait
