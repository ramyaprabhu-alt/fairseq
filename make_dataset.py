from fairseq.data.indexed_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder
import numpy as np
test = MMapIndexedDataset("/ramyapra/fairseq/data-bin/wikitext-103/test")
# dataset = MMapIndexedDatasetBuilder("newtest_final.bin", dtype=np.uint16)
# for i in range(500):
#     dataset.add_item(test[i])
# dataset.finalize("newtest_final.idx")
l = []
for i in test:
    for j in i:
        l.append(int(j))

print(l)