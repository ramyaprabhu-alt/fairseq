DATA_PATH=/ramyapra_/fairseq/data-bin/wikitext-103
MODEL_PATH=/ramyapra_/en_moe_lm_15b/model.pt
python -m fairseq_cli.eval_lm $DATA_PATH\
  --path $MODEL_PATH \
  --gen-subset newtest \
  --sample-break-mode none \
  --tokens-per-sample 1 \
  --batch-size 1 \
  --fp16 \
  --output-word-probs \
  --is-moe \
  --distributed-world-size 4 \
  --model-overrides "{'world_size': 4, 'moe_eval_capacity_token_fraction': 0.05}"
