import sys
import data_utils
import tensorflow as tf
import numpy as np
from network import Network
from tensorflow.contrib.rnn import LSTMBlockCell


""" Selecting the adequate experiment and checkpoint file to evaluate based on arguments fed to the program """
        
# Default paremeters for experiment A
LSTM_SIZE = 512
EXPERIMENT = 'A'
CHECKPOINT_FILE = None

if len(sys.argv) < 2:
    print("Running experiment A")
    CHECKPOINT_FILE = input("Choose a checkpoint_file to evaluate:")
    
elif len(sys.argv) < 3:
    arg = sys.argv[1]
    if arg == "A":
        EXPERIMENT = 'A'
        CHECKPOINT_FILE = input("Choose a checkpoint_file to evaluate:")
        print("Running experiment A on file " + CHECKPOINT_FILE)
    elif arg == "B":
        EXPERIMENT = 'B'
        CHECKPOINT_FILE = input("Choose a checkpoint_file to evaluate:")
        print("Running experiment B on file " + CHECKPOINT_FILE)
    elif arg == "C":
        LSTM_SIZE = 1024
        EXPERIMENT = 'C'
        CHECKPOINT_FILE = input("Choose a checkpoint_file to evaluate:")
        print("Running experiment C on file " + CHECKPOINT_FILE)
    else:
        CHECKPOINT_FILE = arg
        print("Running experiment A on file " + CHECKPOINT_FILE)

elif len(sys.argv) < 4:
    exp = sys.argv[1]
    CHECKPOINT_FILE = sys.argv[2]
    if exp != "A" and exp != "B" and exp != "C" :
        EXPERIMENT = 'A'
        print("Running experiment A on file " + CHECKPOINT_FILE)
    else:
        if exp == "C":
            LSTM_SIZE = 1024
        print("Running experiment " + exp + " on file " + CHECKPOINT_FILE)
        EXPERIMENT = exp
    
    
"""Flags representing constants of our project """    
 
# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "data/processed/sentences_test.npy",
                       "Path to the test data. This data should be distinct from the training data.")
tf.flags.DEFINE_string("group_number", "19", "Our group number")
# Test parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/" + CHECKPOINT_FILE + "/checkpoints/", "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Parameters
tf.flags.DEFINE_integer("default_lstm_size", 512, "default LSTM Size")
tf.flags.DEFINE_integer("lstm_size", LSTM_SIZE, "LSTM Size (default: 512)")
tf.flags.DEFINE_boolean("enable_dropout", False, "Enable the dropout layer")
tf.flags.DEFINE_boolean("word2Vec", False, "True if word2Vec embeddings should be used")
tf.flags.DEFINE_integer("sentence_len", 30, "Length of sentence")
tf.flags.DEFINE_integer("vocab_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embeddings")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

""" Processing of testing data"""

# Load data
print("Loading and preprocessing test dataset \n")
x_test = np.load(FLAGS.data_file_path).astype(np.int32) 
shuffled_indices = np.random.permutation(len(x_test))
x_shuffled = x_test

vocab = np.load("data/processed/sentences.train_vocab.npy")  # vocab contains [symbol: id]
vocabLookup = dict((v,k) for k,v in vocab.item().items()) # flip our vocab dict so we can easy lookup [id: symbol]

""" Evaluating model"""

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    
    with sess.as_default():
        next_sentence = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.sentence_len]) # [batch size, sentence_len]
        dataset = tf.data.Dataset.from_tensor_slices(next_sentence).batch(FLAGS.batch_size)
        # # create the iterator
        iter = dataset.make_initializable_iterator() # create the iterator
        next_batch = iter.get_next()
        #Creating the model
        network = Network(sess, next_batch, None, load_embeddings=FLAGS.word2Vec, sentence_len=30, calculate_loss=True)
        # Restore the variables without loading the meta graph!
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        # Generate batches for one epoch
        batches = data_utils.batch_iter(list(x_shuffled), FLAGS.batch_size, 1, shuffle=False)
        
        # Collect the predictions here
        all_predictions = np.array([], dtype=np.int64).reshape(0, 30)
        perplexities = []
        
        print("Starting perplexity computation")
        counter = 0
        for word_inputs in batches:
            sess.run(iter.initializer, feed_dict={
                next_sentence: word_inputs
            })

            probs = sess.run([network.next_words_probs])
            perp = (data_utils.calc_perplexity(probs, word_inputs[:, 1:]))[0]
            perplexities.append(perp)
            print(counter, "/", len(x_test), perp)
            counter += 1


avg = np.average(perplexities)
print(f"Average: {avg}")

"""Writting the perplexities to output file """
with open("group" + FLAGS.group_number + "perplexity" + EXPERIMENT, 'w') as f:
    for i in range(len(perplexities)):
        f.write(str(perplexities[i]) +"\n")
    
