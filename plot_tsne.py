import numpy as np
import matplotlib.pyplot as plt

vocab_path = "./output/vocab/vocab.txt"
tsne_path = "./output/vocab/tsne.txt"
scale_factor = 100000

with open(tsne_path, "r") as f:
  data_list = []
  for line in f:
    x, y = line.split("\t")
    data_list.append([float(x) * scale_factor, float(y) * scale_factor])
  data = np.asarray(data_list, dtype=np.float32)


vocab_dict = {}
with open(vocab_path, "r") as f:
  for line in f:
    word, index = line.split(" ")
    vocab_dict[int(index)] = word

N = data.shape[0]
labels = [('%s' % vocab_dict[i]) for i in range(N)]
# plt.subplots_adjust(bottom = 0.1)

plt.scatter(data[:200, 0], data[:200, 1], color='b', alpha=0, s=1)

for label, x, y in zip(labels[:200], data[:200, 0], data[:200, 1]):
  # plt.text(x, y, label)
  plt.annotate(
      label,
      xy = (x, y), fontsize=2)

# plt.show()
plt.savefig('./output/vocab/test.pdf', dpi=1200, format='pdf')