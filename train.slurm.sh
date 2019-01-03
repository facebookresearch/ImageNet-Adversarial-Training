#!/bin/bash -e
#SBATCH --output=logs/job-%j.%N.out
#SBATCH --error=logs/job-%j.%N.err
#SBATCH --ntasks-per-node=8  # 8 tasks per node
#SBATCH --gres=gpu:volta:8		 # 8 GPUs per node
#SBATCH --cpus-per-task=10   # 80/8 cpus per task
#SBATCH --mem=200G

# To run on 4 nodes x 8 GPUs: use "mkdir -p logs && sbatch --nodes=4 slurm.script"

echo "NNODES: $SLURM_NNODES"
echo "JOBID: $SLURM_JOB_ID"
env | grep PATH

export TENSORPACK_PROGRESS_REFRESH=20
export TENSORPACK_SERIALIZE=msgpack

DATA_PATH=~/data/imagenet
BATCH=32
CONFIG=$1

# launch data
srun --output=logs/data-%J.%N.log \
		 --error=logs/data-%J.%N.err \
		 --gres=gpu:0 --cpus-per-task=60 --mincpus 60 \
	--ntasks=$SLURM_NNODES --ntasks-per-node=1 \
	python ./serve-data.py --data $DATA_PATH --batch $BATCH &
DATA_PID=$!

# launch training
# https://www.open-mpi.org/faq/?category=openfabrics#ib-router has document on IB options
# the queue parameters sometimes can hang the communication (for some MPI versions and some operations)
	#-mca btl tcp,self \
mpirun -output-filename logs/train-$SLURM_JOB_ID.log -tag-output \
	-bind-to none -map-by slot \
	-mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
	-x NCCL_IB_CUDA_SUPPORT=1 -x NCCL_IB_DISABLE=0 -x NCCL_DEBUG=INFO \
	python ./train.py --data $DATA_PATH --batch $BATCH $CONFIG &
MPI_PID=$!

wait $MPI_PID
