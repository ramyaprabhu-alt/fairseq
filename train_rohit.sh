export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/home/t-rohitd/fairseq"

############### CHANGE STUFF WITHIN THESE LINES ###############
NODES=4
GPUS_PER_NODE=16

NUM_EXPERTS=512
TOKENS_PER_SAMPLE=1024
BATCH_SIZE=8 # batch size per GPU
GRAD_ACC=1 # gradient accumulation
############### CHANGE STUFF WITHIN THESE LINES ###############

TOTAL_GPUS=$(($NODES * $GPUS_PER_NODE))
TOTAL_BATCH_SIZE=$(($BATCH_SIZE * $GRAD_ACC * $TOTAL_GPUS * $TOKENS_PER_SAMPLE))
LR=`python3 -c "import math; print(3e-4 * math.sqrt($TOTAL_BATCH_SIZE / 500000.0))"` # scale LR as needed
TOTAL_UPDATES=`python3 -c "import math; print(math.ceil(300e9/$TOTAL_BATCH_SIZE))"`  # train for 300B tokens
WARMUP_UPDATES=`python3 -c "import math; print(math.ceil(375e6/$TOTAL_BATCH_SIZE))"` # warmup for 375M tokens

if [ $(( $BATCH_SIZE % 3 )) -eq 0 ]; then
     REQUIRED_BATCH_SIZE_MULTIPLE=$(($BATCH_SIZE / 3))
elif [ $(( $BATCH_SIZE % 2 )) -eq 0 ]; then
     REQUIRED_BATCH_SIZE_MULTIPLE=$(($BATCH_SIZE / 2))
else
     REQUIRED_BATCH_SIZE_MULTIPLE=$BATCH_SIZE
fi

echo "Effective batch size: $TOTAL_BATCH_SIZE"
echo "Corrected learning rate: $LR"
echo "Total steps: $TOTAL_UPDATES"
echo "Warmup steps: $WARMUP_UPDATES"
echo "Required batch size multiple: $REQUIRED_BATCH_SIZE_MULTIPLE"

set -ux
SRC_DIR=/mnt/gandiva_blob/MEGATRON_FINAL_STUFF/c4_processed_merged/
CKPT_DIR=~/local_checkpoint_dir/

# launch the job (adjust port and --cpu-bind if needed)
DISTRIBUTED_PORT=12345
srun -o ${CKPT_DIR}/train_log.txt --gpus-per-node ${GPUS_PER_NODE} --ntasks-per-node ${GPUS_PER_NODE} --cpus-per-task 6 --nodes $NODES --mem-per-gpu 80G \
  python fairseq_cli/train.py \
  --train-subset train_shifted \
  --distributed-port ${DISTRIBUTED_PORT} \
  --save-dir ${CKPT_DIR} --save-interval-updates 2000 --save-async \
  --load-checkpoint-on-all-dp-ranks --checkpoint-shard-count ${TOTAL_GPUS} \
  --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
  --task language_modeling ${SRC_DIR} --tokens-per-sample ${TOKENS_PER_SAMPLE} \
  --arch transformer_lm_gpt2_small --share-decoder-input-output-embed \
  --decoder-layers 24 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
  --decoder-attention-heads 16 \
  --moe-expert-count ${NUM_EXPERTS} --moe-freq 2 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 --num-workers-valid 0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr ${LR} --lr-scheduler linear --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size ${BATCH_SIZE} --update-freq ${GRAD_ACC} --required-batch-size-multiple ${REQUIRED_BATCH_SIZE_MULTIPLE} \
  --max-update ${TOTAL_UPDATES} --disable-validation \
  --log-format json --log-interval 50

  # For restoring: --restore-file ${CKPT_DIR}/checkpoint_1_10000.pt --load-checkpoint-on-all-dp-ranks --checkpoint-shard-count 96 \
  # Change the restore checkpoint number
