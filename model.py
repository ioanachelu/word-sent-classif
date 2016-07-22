import tensorflow as tf
import numpy as np
import config
import time
import os

class SentenceCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, sess):
      self.sequence_length = sequence_length
      self.num_classes = num_classes
      self.vocab_size = vocab_size
      self.sess = sess

      # Placeholders for input, output and dropout
      self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
      self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Keeping track of l2 regularization loss (optional)
      self.l2_loss = tf.constant(0.0)


    def inference(self):
      # Embedding layer
      with tf.device('/cpu:0'), tf.name_scope("w_embedding"):
        embed = np.load(config.word2vec_embed_path)
        embed = embed.astype(np.float32)
        self.W_embed = tf.Variable(embed, name="W_embed")
        # self.W_embed = tf.to_float(W_embed, name='ToFloat')
        # W = tf.Variable(
        #     tf.random_uniform([self.vocab_size, config.embedding_size], -1.0, 1.0),
        #     name="W")
        # self.W_embed = tf.cast(W_embedding, tf.float32_ref)
        self.embedded_seq = tf.nn.embedding_lookup(self.W_embed, self.input_x)
        self.embedded_seq_expanded = tf.expand_dims(self.embedded_seq, -1)


      # Create a convolution + maxpool layer for each filter size
      pooled_outputs = []
      for i, filter_size in enumerate(config.filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
          filter_shape = [filter_size, config.embedding_size_kernel, 1, config.num_filters]
          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")
          conv = tf.nn.conv2d(
              self.embedded_seq_expanded,
              W,
              strides=[1, 1, 1, 1],
              padding="VALID",
              name="conv")
          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
          # Maxpooling over the outputs
          pooled = tf.nn.max_pool(
              h,
              ksize=[1, self.sequence_length - filter_size + 1, config.embedding_size - config.embedding_size_kernel + 1, 1],
              strides=[1, 1, 1, 1],
              padding='VALID',
              name="pool")
          pooled_outputs.append(pooled)

      # Combine all the pooled features
      num_filters_total = config.num_filters * len(config.filter_sizes)
      self.h_pool = tf.concat(3, pooled_outputs)
      self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

      # Add dropout
      with tf.name_scope("dropout"):
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

      # Final (unnormalized) scores and predictions
      with tf.name_scope("fc"):
        W = tf.get_variable(
            "W_fc",
            shape=[num_filters_total, self.num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b_fc")
        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

      # CalculateMean cross-entropy loss
      with tf.name_scope("cross-entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
        self.loss = tf.reduce_mean(losses) + config.l2_reg_lambda * self.l2_loss
        self.val_loss_average = tf.train.ExponentialMovingAverage(0.9999, name='val_loss_mov_avg')
        self.val_loss_average_op = self.val_loss_average.apply([self.loss])

      # Accuracy
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def train(self):
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

      with tf.control_dependencies([self.val_loss_average_op]):
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


    def summary(self):
      # Keep track of gradient values and sparsity (optional)
      grad_summaries = []
      for grad, var in self.grads_and_vars:
        if grad is not None:
          grad_hist_summary = tf.histogram_summary(var.op.name + '/gradients/hist', grad)
          sparsity_summary = tf.scalar_summary(var.op.name + '/gradients/sparsity', tf.nn.zero_fraction(grad))
          grad_summaries.append(grad_hist_summary)
          grad_summaries.append(sparsity_summary)

      grad_summaries_merged = tf.merge_summary(grad_summaries)

      # Output directory for models and summaries
      timestamp = str(int(time.time()))
      print("Writing to %s\n" % config.out_dir)

      # Summaries for loss and accuracy
      loss_summary = tf.scalar_summary("loss", self.loss)
      acc_summary = tf.scalar_summary("accuracy", self.accuracy)

      # Train Summaries
      self.train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
      train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
      self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, self.sess.graph_def)

      # Dev summaries
      self.val_summary_op = tf.merge_summary([loss_summary, acc_summary])
      val_summary_dir = os.path.join(config.out_dir, "summaries", "val")
      self.val_summary_writer = tf.train.SummaryWriter(val_summary_dir, self.sess.graph_def)
