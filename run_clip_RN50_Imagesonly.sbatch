#!/bin/bash 

#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=v100-32:4
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=finish_train_network
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mnk2978@nyu.edu

module purge;
set -x

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export CUDA_VISIBLE_DEVICES=0,1,2,3
#comments

#trained on iNaturalist with captions
#run command
srun --accel-bind=v \
    /bin/bash /ocean/projects/cis220069p/mnk2989/setup_singularity.bash \
    /bin/bash -c \
    'export WANDB_API_KEY=983121974971d75792b8a5ffdd8058944431e1f2; export PYTHONPATH="$PYTHONPATH:/ocean/projects/cis220069p/mnk2989/vlhub-forked/src"; python /ocean/projects/cis220069p/mnk2989/vlhub-forked/src/training/main.py  \
    --train-data="/ocean/projects/cis220069p/mnk2989/vlhub-forked/metadata/inat2021_with_labels.csv" \
    --report-to wandb \
    --csv-separator "," \
    --inat2021 "/ocean/projects/cis220069p/mnk2989/data/datasets/" \
    --zeroshot-frequency=1  \
    --integer-labels  \
    --csv-caption-key title \
    --save-frequency 4 \
    --warmup 2000 \
    --batch-size=128 \
    --precision=fp32 \
    --epochs=32 \
    --resume "/ocean/projects/cis220069p/mnk2989/logs/2022_12_03-02_02_56-model_RN50-imagesonly-lr_0.0005-b_256-j_4-p_fp32/checkpoints/epoch_28.pt" \
    --workers=4 --model=RN50-imagesonly --local-loss --gather-with-grad --ds-filter inat_classnames'
