import config
import word2vec
import numpy as np

vocab_path = "./output/vocab/vocab.txt"
word2vec_embed_path = "./output/vocab/embed_subsample_glove.npy"
input_path = "/media/ioana/E9A4-CCE4/glove.840B.300d.txt"

def main():
  vocab_dict_inv = {}
  with open(vocab_path, "r") as f:
    for line in f:
      word, index = line.split(" ")
      vocab_dict_inv[word] = int(index)

  W_embed = np.random.uniform(-1.0, 1.0, size=(len(vocab_dict_inv), config.word2vec_embed_size))
  with open(input_path, 'r') as f:
    for line in f:
      vals = line.rstrip().split(' ')
      if vals[0] in vocab_dict_inv.keys():
        W_embed[vocab_dict_inv[vals[0]]] = [float(x) for x in vals[1:]]

  np.save(word2vec_embed_path, W_embed)

if __name__ == '__main__':
  main()