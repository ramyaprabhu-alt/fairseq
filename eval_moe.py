import subprocess

# for i in range(6):
#     subprocess.run(['python', '-m', 'fairseq_cli.run', '/ramyapra/fairseq/data-bin/wikitext-103/', '--memory-efficient-fp16', '--tokens-per-sample', '1024', '--arch', 'transformer_lm_gpt', '--decoder-layers', '1', '--decoder-embed-dim', '12288', '--decoder-ffn-embed-dim', '49152', '--decoder-attention-heads', '96', '--batch-size', '16', '--moe-expert-count', str(2**i),'--moe-freq', '1', '--moe-gating-use-fp32', '--moe-second-expert-policy', 'all', '--moe-eval-capacity-token-fraction', str(2/(2**i))])


for i in range(6):
    for j in range(5):
        for k in [256, 512, 1024, 2048]:
            subprocess.run(['python', '-m', 'fairseq_cli.run', '/ramyapra/fairseq/data-bin/wikitext-103/', '--memory-efficient-fp16', '--tokens-per-sample', str(k), '--arch', 'transformer_lm_gpt', '--decoder-layers', '1', '--decoder-embed-dim', '12288', '--decoder-ffn-embed-dim', '49152', '--decoder-attention-heads', '96', '--batch-size', str(2**j), '--moe-expert-count', str(2**i),'--moe-freq', '1', '--moe-gating-use-fp32', '--moe-second-expert-policy', 'all', '--moe-eval-capacity-token-fraction', str(-1)])