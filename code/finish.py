import data_utils
import tensorflow as tf
import numpy as np
from network import Network
import sys


""" Selecting the model to use to generate sentence continuation """

CHECKPOINT_FILE = None
if len(sys.argv) < 2:
    CHECKPOINT_FILE = input("Choose a checkpoint_file to use:")

else:
    CHECKPOINT_FILE = sys.argv[1]

"""Flags representing constants of our project """   
  
# Data loading parameters
tf.flags.DEFINE_string("output_file_path", "data/processed/group19.continuation",
                       "Path to the continuation data. This data should be distinct from the training data.")
tf.flags.DEFINE_string("data_file_path", "data/processed/sentences.continuation.npy",
                       "Path to the continuation data. This data should be distinct from the training data.")
tf.flags.DEFINE_string("vocab_file_path", "data/processed/sentences.train_vocab.npy",
                       "Path to the vocab continuation data. This data should be distinct from the training data.")

#Model parameters
tf.flags.DEFINE_integer("lstm_size", 1024, "LSTM Size (Default: 1024")
tf.flags.DEFINE_boolean("word2Vec", False, "True if word2Vec embeddings should be used")
tf.flags.DEFINE_integer("continuation_max_length", 20,
                       "Maximum length of the continuation sentence (default: 20, from the project description)")
tf.flags.DEFINE_integer("default_lstm_size", 512, "default LSTM Size")
tf.flags.DEFINE_integer("vocab_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embeddings")


# Test parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/" + CHECKPOINT_FILE + "/checkpoints/", "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Dropout params
tf.flags.DEFINE_boolean("enable_dropout", False, "Enable the dropout layer (default: True)")
tf.flags.DEFINE_float('keep_prob', 0.5, 'dropout rate')

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

""" Processing of input data"""

# Load data
print("Loading and preprocessing test dataset \n")
x_test = np.load(FLAGS.data_file_path)  # data_utils.load_data_and_labels_test(FLAGS.data_file_path, FLAGS.past_words)

# Load vocab
vocab = np.load(FLAGS.vocab_file_path)  # vocab contains [symbol: id]
vocabLookup = dict((v,k) for k,v in vocab.item().items()) # flip our vocab dict so we can easy lookup [id: symbol]

END_OF_SENTENCE = vocab.item()["<eos>"]


checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    
    with sess.as_default():
        # make a dataset from a numpy array
        next_word = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1]) # [batch size, sentence_len]
        dataset = tf.data.Dataset.from_tensor_slices(next_word).batch(FLAGS.batch_size)
        # # create the iterator
        iter = dataset.make_initializable_iterator() # create the iterator
        next_batch = iter.get_next()

        network = Network(sess, next_batch, None, load_embeddings=FLAGS.word2Vec, sentence_len=1, calculate_loss=False)
        # Restore the variables without loading the meta graph!
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        # Tensors we want to evaluate
        predictions = network.predicted_words # fine because its for a single word
        # Generate batches for one epoch
        batches = data_utils.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        """ Start generating end of sentences"""
        sentence_predictions = []
        for x_test_batch in batches:
            
            sentence_start = x_test_batch[0]
            lstm_states = [np.zeros((FLAGS.batch_size, FLAGS.lstm_size)),
                           np.zeros((FLAGS.batch_size, FLAGS.lstm_size)),
                           np.zeros((FLAGS.batch_size, FLAGS.lstm_size)),
                           np.zeros((FLAGS.batch_size, FLAGS.lstm_size))]

            current_output = None
            total_output = []
            sentence_len = 0 # to remove the end

            for word in sentence_start:
                """ 
                Starts by computing hidden state for the start of the sentence
                """
                # feed the start word by word
                if word == END_OF_SENTENCE:
                    break          # Stop at end of sentence of continuation sentence

                sess.run(iter.initializer, feed_dict={
                    next_word: np.resize(np.array([word], dtype=np.int32), (1, 1))
                })
    
                sentence_len += 1
                feed_dict = {
                    network.lstm_c1: lstm_states[0],
                    network.lstm_h1: lstm_states[1],
                    network.lstm_c2: lstm_states[2],
                    network.lstm_h2: lstm_states[3]
                }
                fetch_dict = [
                    predictions,
                    network.states,
                ]
                batch_predictions, states = sess.run(fetch_dict, feed_dict)
                lstm_states = [
                    states[0].c,
                    states[0].h,
                    states[1].c,
                    states[1].h
                ]
                current_output = batch_predictions[0] # batch size = 1


            total_output.append(current_output)
            x = 0
            while current_output != END_OF_SENTENCE and x < FLAGS.continuation_max_length:
                """
                Generates the end of the sentence
                """
                sess.run(iter.initializer, feed_dict={
                    next_word: np.resize(np.array([current_output], dtype=np.int32), (1, 1))
                })
                feed_dict = {
                    network.lstm_c1: lstm_states[0],
                    network.lstm_h1: lstm_states[1],
                    network.lstm_c2: lstm_states[2],
                    network.lstm_h2: lstm_states[3]
                }
                fetch_dict = [
                    predictions,
                    network.states
                ]
                batch_predictions, states = sess.run(fetch_dict, feed_dict)
                lstm_states = [
                    states[0].c,
                    states[0].h,
                    states[1].c,
                    states[1].h
                ]
                current_output = batch_predictions[0]  # batch size = 1
                total_output.append(current_output)
                x += 1

            # output the total sentence
            print(f"Sentence: [{ data_utils.toString(data_utils.makeSymbols(sentence_start[:sentence_len]),vocabLookup)}] { data_utils.toString(data_utils.makeSymbols(total_output, vocabLookup)) }")
            output_nums = sentence_start[1:sentence_len].tolist() + total_output
            sentence_predictions.append(data_utils.toString(data_utils.makeSymbols(output_nums, vocabLookup)))


with open(FLAGS.output_file_path, 'w') as f:
    """
    Outputs the file containing the sentence predictions
    """
    for s in sentence_predictions:
        f.write("%s\n" % s)

