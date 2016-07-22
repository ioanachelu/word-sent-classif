import config
import word2vec
import numpy as np

vocab_path = "./output/vocab/vocab.txt"
word2vec_embed_path = "./output/vocab/embed_subsample.npy"

def main():
  model = word2vec.load(config.word2vec_path, 'auto', 50, None, 'ISO-8859-1')
  vocab_dict = {}
  with open(vocab_path, "r") as f:
    for line in f:
      word, index = line.split(" ")
      vocab_dict[int(index)] = word

  W_embed = np.zeros((len(vocab_dict), config.word2vec_embed_size))
  for i in range(len(vocab_dict)):
    if vocab_dict[i] in model.vocab_hash.keys():
      W_embed[i] = model[vocab_dict[i]][:]
    else:
      W_embed[i] = np.random.uniform(-1.0, 1.0, size=300)
  np.save(word2vec_embed_path, W_embed)


if __name__ == '__main__':
  main()