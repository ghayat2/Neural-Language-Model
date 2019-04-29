import numpy as np
import tensorflow as tf
from gensim import models

def clean_string(string):
    return string.lower()

def toString(array):
    return ' '.join(array)

def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    '''
          session        Tensorflow session object
          vocab          A dictionary mapping token strings to vocabulary IDs
          emb            Embedding tensor of shape vocabulary_size x dim_embedding
          path           Path to embedding file
          dim_embedding  Dimensionality of the external embedding.
        '''

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.item().items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def calc_perplexity(probs, indices):
    """ 
    Calculate the perplexity based on the probabilites of each word in a sentence
    and its indice
    """
   
    eps = 1e-10
    mask = (np.not_equal(3, indices)).astype(int) #Should not count paddings
    sentence_length = np.sum(mask, axis=1)
    return 2**(-np.sum(np.log2(np.maximum(probs[0], eps)) * mask, axis=1)/sentence_length)

def makeSymbols(array, vocabLookup):
    """ 
    Convert array of integers into a sentence based on the dic argument
    """
    return list(vocabLookup[x] for x in array)