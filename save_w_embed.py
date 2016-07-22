import calc_tsne
import config
import os

vocab_path = "./output/vocab/vocab.txt"
embed_path = "./output/vocab/embed.txt"

def main():
  model = calc_tsne.resume_model()
  if model is not None:
     #vocab_size, embed_size
    # vocab_words = []
    # vocab_indices = []
    # with open(vocab_path, "r") as f:
    #   for line in f:
    #     word, index = line.split(" ")
    #     vocab_words.append(word)
    #     vocab_indices.append(int(index))
    embed = model.sess.run(model.W_embed)
    with open(os.path.join(config.out_dir, "vocab/embed.txt"), "wt") as f:
      for result in embed:
        fmt = ''
        for i in range(1, len(result)):
          fmt = fmt + '{}\t'
        fmt = fmt + '{}\n'
        f.write(fmt.format(*result))


if __name__ == "__main__":
  main()