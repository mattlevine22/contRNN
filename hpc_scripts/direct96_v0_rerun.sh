#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "l96_rerun"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/directL96_v0_pi100"
echo "Sending results to $round_dir"
# srun --ntasks=1 python3 train_l96_hpc.py --max_grad_norm 1 --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 10000 --hpc 1 --dim_hidden 500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu500_layers2_gpu_bs10000" &
# srun --ntasks=1 python3 train_l96_hpc.py --max_grad_norm 1 --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu500_layers2_gpu_bs1000"
srun --ntasks=1 python3 train_l96_hpc.py --eval_time_limit 60000 --plot_interval 1 --gpu 0 --lr 0 --epochs 0 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --T_long 15 --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_debug_Tlong15" &
srun --ntasks=1 python3 train_l96_hpc.py --eval_time_limit 60000 --plot_interval 1 --gpu 0 --lr 0 --epochs 0 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --T_long 500 --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_debug_Tlong500" &
srun --ntasks=1 python3 train_l96_hpc.py --eval_time_limit 60000 --plot_interval 1 --gpu 0 --lr 0 --epochs 0 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --T_long 2000 --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_debug_Tlong2000" &
srun --ntasks=1 python3 train_l96_hpc.py --eval_time_limit 60000 --plot_interval 1 --gpu 0 --lr 0 --epochs 0 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --T_long 20000 --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000_debug_Tlong20000"
wait
