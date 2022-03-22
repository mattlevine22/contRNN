#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=7   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "directL96_v0"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="experiments/directL96_v0"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 10000 --hpc 1 --dim_hidden 500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu500_layers2_gpu_bs10000" &
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu500_layers2_gpu_bs1000" &
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 0 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_cpu_bs1000" &
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 2500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu2500_layers2_gpu_bs1000" &
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 0 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 500 --n_layers 2 --activation gelu --output_dir "$round_dir/gelu500_layers2_cpu_bs1000" &
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 1000 --hpc 1 --dim_hidden 500 --n_layers 3 --activation gelu --output_dir "$round_dir/gelu500_layers3_gpu_bs1000" &
srun --ntasks=1 python3 train_l96_hpc.py --plot_interval 10 --gpu 1 --lr 0.01 --epochs 1000 --do_normalization 1 --batch_size 10000 --hpc 1 --dim_hidden 500 --n_layers 3 --activation gelu --output_dir "$round_dir/gelu500_layers3_gpu_bs10000"
wait
