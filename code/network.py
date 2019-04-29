# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import data_utils

FLAGS = tf.flags.FLAGS

class Network:

    def __init__(self, session, batch, vocab, load_embeddings=False, sentence_len=30, calculate_loss=True):
        """
        Sets up the graph of our RNN model.
            session           Tensorflow session object
            batch             The batch of sentences
            vocab             A dictionary mapping token strings to vocabulary IDs
            load_embeddings   Word embeddings our model will use (default: our own)
            sentence_len      The length of  one sentence  (default: 30)
            calculate_loss    True if the model is in training phase (default:True)
        
        """
        self.sentence_len = sentence_len
        self.calculate_loss = calculate_loss
        self.word_inputs = batch
        self.initializer = tf.contrib.layers.xavier_initializer

        if FLAGS.lstm_size > FLAGS.default_lstm_size:
            print(f"Running with downsize layer from {FLAGS.lstm_size} to {FLAGS.default_lstm_size}!")

        with tf.name_scope("embedding"):

            self.embedding_matrix = tf.get_variable("embedding_matrix",
                                                    initializer=tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim], -1.0, 1.0),
                                                    dtype=tf.float32,
                                                    trainable=True)

            if load_embeddings:
                data_utils.load_embedding(session, vocab, self.embedding_matrix, FLAGS.path_embeddings, FLAGS.embedding_dim, FLAGS.vocab_size)

            self.embedded_words = tf.nn.embedding_lookup(self.embedding_matrix,
                                                         self.word_inputs)  # DIM [batch_size, sentence_len, embedding_dim]

        with tf.name_scope("rnn"):
            # Stacked LSTM layers architecture, with 2 layers
            lstms = (tf.contrib.rnn.LSTMBlockCell(num_units=FLAGS.lstm_size, dtype=tf.float32),
                     tf.contrib.rnn.LSTMBlockCell(num_units=FLAGS.lstm_size, dtype=tf.float32))

            with tf.variable_scope('hidden_state'):
                #Placeholders holding the value of the hidden states (default: zero matrix)
                self.lstm_c1 = tf.placeholder_with_default(np.zeros((FLAGS.batch_size, FLAGS.lstm_size), dtype=np.float32),
                                              shape=[None, lstms[0].state_size.c],
                                              name='c1_in')
                self.lstm_h1 = tf.placeholder_with_default(np.zeros((FLAGS.batch_size, FLAGS.lstm_size), dtype=np.float32),
                                              shape=[None, lstms[0].state_size.h],
                                              name='h1_in')
                self.lstm_c2 = tf.placeholder_with_default(np.zeros((FLAGS.batch_size, FLAGS.lstm_size), dtype=np.float32),
                                              shape=[None, lstms[1].state_size.c],
                                              name='c2_in')
                self.lstm_h2 = tf.placeholder_with_default(np.zeros((FLAGS.batch_size, FLAGS.lstm_size), dtype=np.float32),
                                              shape=[None, lstms[1].state_size.h],
                                              name='h2_in')

                state_in1 = tf.contrib.rnn.LSTMStateTuple(self.lstm_c1, self.lstm_h1)
                state_in2 = tf.contrib.rnn.LSTMStateTuple(self.lstm_c2, self.lstm_h2)
                self.states = [state_in1, state_in2]

            # Add a down size matrix if necessary
            if FLAGS.lstm_size > FLAGS.default_lstm_size:
                down_size = tf.get_variable("down_size", [FLAGS.lstm_size, FLAGS.default_lstm_size])

            self.W_h = tf.get_variable("W_h", [FLAGS.default_lstm_size, FLAGS.vocab_size], tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.b_h = tf.get_variable("b_h", [FLAGS.vocab_size], tf.float32, initializer=tf.zeros_initializer())

            self.predictions = []
            self.next_words_probs = []
            self.loss = 0.0
            extra_count_for_predict = int(self.calculate_loss == False)
            
            for i in range(self.sentence_len - 1 + extra_count_for_predict):
                # words = [batch_size, embeddings] #next_words_index = [batch_size]
                words = self.embedded_words[:, i, :]
                # First layer: input word is the actual word
                output, self.states[0] = lstms[0](words, self.states[0])  # output = [batch_size, embedding_dim]
                # add dropout layer
                if FLAGS.enable_dropout:
                    output = tf.nn.dropout(output, rate=1 - FLAGS.keep_prob)
                # Second layer: input word is the prediction from the first layer
                output, self.states[1] = lstms[1](output, self.states[1])
                # add second dropout layer
                if FLAGS.enable_dropout:
                    output = tf.nn.dropout(output, rate=1-FLAGS.keep_prob)
                # Down-project working if necessary
                if FLAGS.lstm_size > FLAGS.default_lstm_size:
                    output = tf.matmul(output, down_size)
                # The output comes from he second layer
                self.logits = tf.matmul(output, self.W_h) + self.b_h  # logits = [batch_size, VOCABULARY_LEN]

                probabilities = tf.nn.softmax(self.logits, name="softmax_probs")  # size = [batch_size, VOCABULARY_LENGTH]

                if self.calculate_loss:
                    next_words_index = self.word_inputs[:, i + 1]
                    # [0, word i + 1 from B0], ... , [63, word i + 1 from B63]
                    indices_of_next_words = tf.stack([tf.range(FLAGS.batch_size), next_words_index], axis=1)
                    # The probas of the words that should have been predicted
                    next_word_prob = tf.gather_nd(probabilities, indices_of_next_words)
                    self.next_words_probs.append(next_word_prob)

                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=next_words_index)
                    self.loss = tf.math.add(self.loss, tf.reduce_mean(losses),
                                            name="cross_entropy_loss") 

                self.predicted_words = tf.argmax(self.logits, axis=1, name="predicted_words")
                self.predictions.append(self.predicted_words)

            if self.calculate_loss:
                self.next_words_probs = tf.stack(self.next_words_probs, axis=1, name="probs")  # [batch_size, sentence_length]
                words = self.word_inputs[:, 1:]  # Skipping <bos>

        # Calculating accuracy
        with tf.name_scope("accuracy"):
            if self.calculate_loss:
                correct_predictions = tf.equal(self.predictions, tf.transpose(tf.cast(self.word_inputs[:, 1:], tf.int64)))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64), name="accuracy")
