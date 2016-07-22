import tensorflow as tf
from bhtsne.bhtsne import bh_tsne
import config
from model import SentenceCNN
import os
import preprocessing as pre
import numpy as np

DEFAULT_THETA = 0.5
EMPTY_SEED = -1
VERBOSE = True
INITIAL_DIMENSIONS = 50

def resume_model():

  x, y, vocabulary, vocabulary_inv = pre.load_data()
  # Randomly shuffle data
  shuffle_indices = np.random.permutation(np.arange(len(y)))
  x_shuffled = x[shuffle_indices]
  y_shuffled = y[shuffle_indices]
  x_train, x_val = x_shuffled[:-1000], x_shuffled[-1000:]
  sess = tf.Session()
  cnn = SentenceCNN(
      sequence_length=x_train.shape[1],
      num_classes=2,
      vocab_size=len(vocabulary),
      sess=sess
  )
  cnn.inference()
  cnn.train()

  # Create a saver.
  saver = tf.train.Saver()

  checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint

    saver.restore(sess, ckpt.model_checkpoint_path)
    cnn.sess = sess
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0,
    # extract global_step from it.

    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

    return cnn
  else:
    print('No checkpoint file found. Cannot resume.')
    return None

def main(_):
  model = resume_model()
  if model is not None:
    norm_w_embed = tf.nn.l2_normalize(model.W_embed, 1) # [vocab_size, embed_size]
    embedings = model.sess.run(norm_w_embed)
    results = bh_tsne(embedings, no_dims=2, perplexity=50, theta=DEFAULT_THETA, randseed=EMPTY_SEED,
              verbose=VERBOSE)

    with open(os.path.join(config.out_dir, "vocab/tsne.txt"), "w") as f:
      for result in results:
        fmt = ''
        for i in range(1, len(result)):
          fmt = fmt + '{}\t'
        fmt = fmt + '{}\n'

        f.write(fmt.format(*result))
  else:
    print "Model is None"


if __name__ == "__main__":
  tf.app.run()