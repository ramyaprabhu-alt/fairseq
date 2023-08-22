set -eux 
DATA_PATH=/ramyapra/fairseq/data-bin/wikitext-103/
MODEL_PATH=/ramyapra/en_dense_lm_125m/model.pt
python -m fairseq_cli.eval_lm $DATA_PATH \
  --path $MODEL_PATH \
  --gen-subset test \
  --sample-break-mode none \
  --tokens-per-sample 1024 \
  --batch-size 1 \
  --fp16 \
  --output-word-probs \
  --distributed-world-size 1 \
  --model-overrides "{'world_size': 1}"
