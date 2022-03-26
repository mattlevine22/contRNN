#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "gpu_L96_bs"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/directL96_v1.3"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 train_l96_hpc.py --patience 5 --factor 0.1 --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 1 --lr 0.01 --epochs 50000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_gpu_bs1000_gradclip1_factor0.1_patience5" &
srun --ntasks=1 python3 train_l96_hpc.py --patience 5 --factor 0.1 --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 1 --lr 0.01 --epochs 50000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 1000 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu1000_layers2_gpu_bs1000_gradclip1_factor0.1_patience5" &
wait
