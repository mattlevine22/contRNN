#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="run6_minlr0.0001"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 1 --multi_traj 1 --lr 0.1 --epochs 200000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_lam1" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 5 --multi_traj 1 --lr 0.1 --epochs 200000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_lam5" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 10 --multi_traj 1 --lr 0.1 --epochs 200000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_lam10" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 1 --multi_traj 1 --lr 0.1 --epochs 200000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_lam1" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 5 --multi_traj 1 --lr 0.1 --epochs 200000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_lam5" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 10 --multi_traj 1 --lr 0.1 --epochs 200000 --window 100 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_lam10" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 200000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_warmup100" &
srun --ntasks=1 python3 test_run_v8_hpc.py --gpu 1 --min_lr 0.0001 --lambda_endpoints 0 --multi_traj 1 --lr 0.1 --epochs 200000 --window 200 --warmup 100 --infer_ic 0 --use_f0 0 --do_normalization 1 --batch_size 500 --hpc 1 --known_inits 0 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs500_warmup100"
wait
