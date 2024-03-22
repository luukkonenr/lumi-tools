#!/bin/bash

#SBATCH --job-name=cl_1_test
#SBATCH --nodes=2
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-00:35:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclude=nid005003,nid007971,nid007972

export HF_HOME=/scratch/project_462000086/risto/transformers_cache

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_TASKS_PER_NODE
export PROCESSES=$SLURM_GPUS_ON_NODE
export MODEL=/scratch/project_462000086/viking-v2/converted_models/viking_v2_33B_iter_0215000_bfloat16/

ln -f -s "${SLURM_JOB_ID}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}.err" logs/latest.err

# compilers in the container
export CC=gcc-10
export CXX=g++-10

mkdir -p workdir
wd=$(realpath workdir)

export PYTHONUSERBASE=/scratch/project_462000086/risto/lumi-tools/container_scripts/pythonuserbase
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif
SING_BIND="/scratch/project_462000086/,$PYTHONUSERBASE,$TRANSFORMERS_CACHE,$wd"
FSDP_TYPE="full_shard"
cat <<EOF >accelerate_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
        fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
        fsdp_backward_prefetch: BACKWARD_PRE
        fsdp_cpu_ram_efficient_loading: true
        fsdp_forward_prefetch: false
        fsdp_offload_params: false
        fsdp_sharding_strategy: FULL_SHARD
        fsdp_state_dict_type: SHARDED_STATE_DICT
        fsdp_sync_module_states: true
        fsdp_use_orig_params: true
    machine_rank: 0
    main_process_ip: $MASTER_ADDR
    main_process_port: 9999
    main_training_function: main
    mixed_precision: bf16
    num_machines: $SLURM_NNODES
    num_processes: $PROCESSES
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
EOF

CMD=" \
    run_clm.py \
    --model_name_or_path $MODEL \
    --output_dir output \
    --overwrite_output_dir \
    --dataset_name wikitext \
    --dataset_config wikitext-2-v1 \
    --do_train \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 1 \
    --max_train_samples 100 \
    --bf16 \
    --bf16_full_eval \
    --block_size 1024 \
    --fsdp $FSDP_TYPE"

if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

echo $WORLD_SIZE
# OTHER LAUNCHERS CAN BE USED HERE
# export LAUNCHER="accelerate launch \
#     --config_file accelerate_config.yaml \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --machine_rank \$SLURM_PROCID \
#     --num_processes $PROCESSES \
#     --num_machines $SLURM_NNODES \
#     "

# CMD="$LAUNCHER $CMD"


# echo "$CMD"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
export RDZV_ID="$SLURM_JOB_ID"
export RDZV_ENDPOINT="$RDZV_HOST:$RDZV_PORT"
export PATH=$PATH:$PYTHONUSERBASE/bin
echo "rdzv endpoint: $RDZV_ENDPOINT"

LAUNCHER="torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node 8 \
    --rdzv_backend=c10d \
    --rdzv_id=$RDZV_ID \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    run_clm.py \
    --model_name_or_path $MODEL \
    --output_dir output \
    --overwrite_output_dir \
    --dataset_name wikitext \
    --dataset_config wikitext-2-v1 \
    --do_train \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --max_train_samples 100 \
    --fp16 \
    --fp16_full_eval \
    --block_size 1024 \
    --fsdp full_shard \
    "
    # --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \"

srun  --label \
    singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B "$SING_BIND" \
    -B "$PWD" \
    "$CONTAINER" \
    bash -c "source /opt/miniconda3/bin/activate pytorch && $LAUNCHER"

