set -eux 
DATA_PATH=/ramyapra/fairseq/data-bin/wikitext-103/
MODEL_PATH=/ramyapra/en_moe_lm_15b/model.pt

python -m fairseq_cli.generate $DATA_PATH \
  --path $MODEL_PATH \
  --gen-subset test \
  --batch-size 2 \
  --fp16 \
  --is-moe \
  --distributed-world-size 1 \
  --ddp-backend pytorch_ddp\
  --model-overrides "{'world_size': 1, 'moe_eval_capacity_token_fraction': 0.05}" \
  --task language_modeling
