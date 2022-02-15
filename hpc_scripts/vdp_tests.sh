#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=9   # number of processor cores (i.e. tasks)
#SBATCH --nodes=9   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "test0"   # job name

module load python3/3.7.0
module load gcc/9.2.0
#conda init bash
#conda activate fno

round_dir="preliminaries"
echo "Sending results to $round_dir"
srun --ntasks=1 python3 vdp_v8_hpc.py --plot_interval 100 --epochs 100000 --window 100 --batch_size 100 --hpc 1 --known_inits 1 --output_dir "$round_dir/small_gelu_bs100_multiTraj_knownIC_Win100" &
srun --ntasks=1 python3 vdp_v8_hpc.py --plot_interval 100 --epochs 100000 --window 250 --batch_size 100 --hpc 1 --known_inits 1 --output_dir "$round_dir/small_gelu_bs100_multiTraj_knownIC_Win250" &
srun --ntasks=1 python3 vdp_v8_hpc.py --plot_interval 100 --epochs 100000 --window 1000 --batch_size 100 --hpc 1 --known_inits 1 --output_dir "$round_dir/small_gelu_bs100_multiTraj_knownIC_Win1000" &
srun --ntasks=1 python3 vdp_v8_hpc.py --lambda_endpoints 0 --plot_interval 100 --epochs 100000 --window 100 --batch_size 100 --hpc 1 --known_inits 0 --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC_Win100" &
srun --ntasks=1 python3 vdp_v8_hpc.py --lambda_endpoints 0 --plot_interval 100 --epochs 100000 --window 250 --batch_size 100 --hpc 1 --known_inits 0 --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC_Win250" &
srun --ntasks=1 python3 vdp_v8_hpc.py --lambda_endpoints 0 --plot_interval 100 --epochs 100000 --window 1000 --batch_size 100 --hpc 1 --known_inits 0 --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC_Win1000" &
srun --ntasks=1 python3 vdp_v8_hpc.py --lambda_endpoints 0.01 --plot_interval 100 --epochs 100000 --window 100 --batch_size 100 --hpc 1 --known_inits 0 --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC_Win100_lambdaEnd0.01" &
srun --ntasks=1 python3 vdp_v8_hpc.py --lambda_endpoints 0.1 --plot_interval 100 --epochs 100000 --window 250 --batch_size 100 --hpc 1 --known_inits 0 --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC_Win250_lambdaEnd0.1" &
srun --ntasks=1 python3 vdp_v8_hpc.py --lambda_endpoints 1 --plot_interval 100 --epochs 100000 --window 1000 --batch_size 100 --hpc 1 --known_inits 0 --output_dir "$round_dir/small_gelu_bs100_multiTraj_UNknownIC_Win1000_lambdaEnd1"
wait
