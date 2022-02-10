#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=8   # number of nodes
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="run2_known_ICs_GPUs"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 0 --do_normalization 0 --hpc 1 --known_inits 1 --use_bilinear 1 --n_layers 1 --output_dir "$round_dir/1LayerBilinear" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 0 --do_normalization 1 --hpc 1 --known_inits 1 --use_bilinear 1 --n_layers 1 --output_dir "$round_dir/1LayerBilinear_Normalized" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 0 --do_normalization 1 --batch_size 1 --hpc 1 --known_inits 1 --dim_hidden 100 --n_layers 3 --output_dir "$round_dir/big_relu_bs1" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 0 --do_normalization 1 --batch_size 10 --hpc 1 --known_inits 1 --dim_hidden 100 --n_layers 3 --output_dir "$round_dir/big_relu_bs10" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 100 --n_layers 3 --output_dir "$round_dir/big_relu_bs100" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 0 --do_normalization 1 --batch_size 1000 --hpc 1 --known_inits 1 --dim_hidden 100 --n_layers 3 --output_dir "$round_dir/big_relu_bs1000" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 1 --do_normalization 1 --hpc 1 --known_inits 1 --dim_hidden 100 --n_layers 3 --output_dir "$round_dir/big_relu_hybrid" &
srun --ntasks=1 python3 test_run_v6_hpc.py --gpu 1 --use_f0 1 --do_normalization 1 --hpc 1 --known_inits 1 --dim_hidden 1000 --n_layers 3 --output_dir "$round_dir/bigger_relu_hybrid"
wait
