#!/usr/bin/env python
import sys
from network import Network
import tensorflow as tf
import numpy as np
import os
import time
import datetime


""" Selecting the adequate experiment based on arguments fed to the program """

# Default paremeters for experiment A
LOAD_EMBEDDINGS = False
LSTM_SIZE = 512

if len(sys.argv) < 2:
    print("Running experiment A")
else:
    exp = sys.argv[1]
    if exp == "B":
        print("Running experiment B")
        LOAD_EMBEDDINGS = True
    elif exp == "C":
        print("Running experiment C")
        LSTM_SIZE = 1024
        LOAD_EMBEDDINGS = True

    else:
        print("Running experiment A")


"""Flags representing constants of our project """     
   
# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "data/processed/sentences.train.npy", "Path to the training data")
tf.flags.DEFINE_string("validation_file_path", "data/processed/sentences.eval.npy", "Path to the eval data")

# Model parameters
tf.flags.DEFINE_integer("sentence_len", 30, "Length of sentence")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embeddings")
tf.flags.DEFINE_boolean("word2Vec", LOAD_EMBEDDINGS, "True if word2Vec embeddings should be used")
tf.flags.DEFINE_integer("vocab_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_string("path_embeddings", "data/wordembeddings-dim100.word2vec", "Path to the word2vec embeddings")

# Training parameters
tf.flags.DEFINE_integer("shuffle_buffer_size", 100, "Buffer size from which the next element will be uniformly chosen from")
tf.flags.DEFINE_integer("repeat_train_set", 8, "Amount of repetitions for the train set")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("default_lstm_size", 512, "default LSTM Size")
tf.flags.DEFINE_integer("lstm_size", LSTM_SIZE, "LSTM Size (default: 512)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this number of steps")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Saves model after this number of steps")
tf.flags.DEFINE_integer("output_prediction_every", 100, "Outputs the real and predicted sentences after this number of steps")
tf.flags.DEFINE_integer("num_checkpoints", 8, "Number of checkpoints to store")
tf.flags.DEFINE_boolean("enable_dropout", True, "Enable the dropout layer")
tf.flags.DEFINE_float('keep_prob', 0.3, 'Retention rate, dropout rate = 1 - keep_prob')

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# for running on EULER
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 2,
                        "TF nodes that perform blocking operations are enqueued on a pool of "
                        "inter_op_parallelism_threads available in each process (default 0).")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 2,
                        "The execution of an individual op (for some op types) can be parallelized on a pool of "
                        "intra_op_parallelism_threads (default: 0).")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))


""" Processing of input data"""

print("Loading and preprocessing training and dev datasets \n")

# Loading vocab
vocab = np.load("data/processed/sentences.train_vocab.npy")  # vocab contains [symbol: id]
vocabLookup = dict((v,k) for k,v in vocab.item().items()) # flip our vocab dict so we can easy lookup [id: symbol]

x = np.load(FLAGS.data_file_path)
validation = np.load(FLAGS.validation_file_path)

# Randomly shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(len(x))
shuffled_validation_indices = np.random.permutation(len(validation))

x_shuffled = x[shuffled_indices]
validation_shuffled = validation[shuffled_validation_indices]

#Shuffled input training dataset
x_train = x_shuffled.astype(np.int32)
#Shuffled input validation dataset
x_dev = validation_shuffled.astype(np.int32)

with tf.Graph().as_default():
    
    """Model and training procedure defintion"""

    sentence_placeholder = tf.placeholder(tf.int32, shape=[None, FLAGS.sentence_len])
    handle = tf.placeholder(tf.string, shape=[])

    train_dataset = tf.data.Dataset.from_tensor_slices(sentence_placeholder)\
        .batch(FLAGS.batch_size, drop_remainder=True)\
        .repeat(FLAGS.repeat_train_set)\
        .shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(sentence_placeholder)\
        .repeat(FLAGS.repeat_train_set * FLAGS.evaluate_every)\
        .batch(FLAGS.batch_size, drop_remainder=True)

    #Iterators on the training and validation dataset
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    iter = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    next_batch = iter.get_next()
    
    train_init_op = iter.make_initializer(train_dataset, name='train_dataset')
    test_init_op = iter.make_initializer(test_dataset, name='test_dataset')

    print("Done")

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    sess = tf.Session(config=session_conf)
    
    with sess.as_default():
        
        # Initialize model
        network = Network(sess, next_batch, vocab, load_embeddings = FLAGS.word2Vec)

        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        # initialise iterators
        sess.run(train_iterator.initializer, feed_dict={sentence_placeholder: x_train})
        sess.run(test_iterator.initializer, feed_dict={sentence_placeholder: x_dev})

        # Defining AdamOptimizer (with gradient clipping)
        global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()
            with tf.name_scope("clip"):
                gradients = optimizer.compute_gradients(network.loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, 5), var) for gradient, var in gradients]
            train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)
        
        """ Writting checkpoints and summaries""" 
        
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", network.loss)
        acc_summary = tf.summary.scalar("accuracy", network.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory 
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        # Define training step
        def train_step():
            """
            A single training step
            """
            
            fetches = [train_op, global_step, train_summary_op, network.loss, network.accuracy, network.word_inputs]
            _, step, summaries, loss, accuracy, words = sess.run(fetches, feed_dict={ handle: train_handle })
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        # Define evaulation step
        def dev_step(writer=None):
            """
            A single evaluation step
            """

            fetches = [global_step, dev_summary_op, network.loss, network.accuracy]
            step, summaries, loss, accuracy = sess.run(fetches, feed_dict={ handle: test_handle })
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        """ Training loop - default option is that the model trains until an OutOfRange exception """           
        current_step = 0
        while True:
            try:
                train_step()
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            except tf.errors.OutOfRangeError:
                print("Iterator of range! Terminating")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                break
