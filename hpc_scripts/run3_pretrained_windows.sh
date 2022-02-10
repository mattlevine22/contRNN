#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --nodes=6   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

#this study takes a model pre-trained on T=1 and re-trains it on T=2,  5, 10
# model used is from /groups/astuart/mlevine/contRNN/l63/run3_smallerModels_known_ICs/small_gelu_bs100/rnn.pt
# taken at approximately epoch 1000 (copied while it was running).
round_dir="run3_smallerModels_known_ICs"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 test_run_v6_hpc.py --lr 0.001 --window 200 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_pre_T2_lr0.001" &
srun --ntasks=1 python3 test_run_v6_hpc.py --lr 0.001 --window 500 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_pre_T5_lr0.001" &
srun --ntasks=1 python3 test_run_v6_hpc.py --lr 0.001 --window 1000 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_pre_T10_lr0.001" &
srun --ntasks=1 python3 test_run_v6_hpc.py --lr 0.0001 --window 200 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_pre_T2_lr0.0001" &
srun --ntasks=1 python3 test_run_v6_hpc.py --lr 0.0001 --window 500 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_pre_T5_lr0.0001" &
srun --ntasks=1 python3 test_run_v6_hpc.py --lr 0.0001 --window 1000 --use_f0 0 --do_normalization 1 --batch_size 100 --hpc 1 --known_inits 1 --dim_hidden 50 --n_layers 2 --activation gelu --output_dir "$round_dir/small_gelu_bs100_pre_T10_lr0.0001"
wait
