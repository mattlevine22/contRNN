#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "directL96_v0"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/directL96_v1.1"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 0 --plot_interval 100 --gpu 0 --lr 0.01 --epochs 300 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_noGradClip" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.01 --patience 5 --epochs 300 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_gradclip1_patience5" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.01 --epochs 300 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_gradclip1" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.01 --epochs 300 --do_normalization 1 --batch_size 5000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs5000_gradclip1" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.01 --epochs 300 --do_normalization 1 --batch_size 10000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs10000_gradclip1" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.1 --epochs 300 --do_normalization 1 --batch_size 10000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs10000_gradclip1_lr0.1" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.01 --epochs 300 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 5000 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu5000_layers2_cpu_bs1000_gradclip1" &
srun --ntasks=1 python3 train_l96_hpc.py --T_long 500 --eval_time_limit 6000 --max_grad_norm 1 --plot_interval 100 --gpu 0 --lr 0.01 --epochs 300 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 3 --activation gelu --output_dir "$round_dir/gelu2500_layers3_cpu_bs1000_gradclip1"
wait
