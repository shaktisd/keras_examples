'''
LSTM Score
Train on 20000 samples, validate on 5000 samples
Epoch 1/2
20000/20000 [==============================] - 1171s - loss: 0.6399 - acc: 0.6315 - val_loss: 0.6224 - val_acc: 0.6768
Epoch 2/2
20000/20000 [==============================] - 814s - loss: 0.5558 - acc: 0.7214 - val_loss: 0.4330 - val_acc: 0.8038

Testing LSTM model
'''


from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
import keras

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove/'
TEXT_DATA_DIR = BASE_DIR + '/test_movie_reviews/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            i = t.find('\n\n')  # skip header
            if 0 < i:
                t = t[i:]
            texts.append(t)
            f.close()
            labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


x_test = data
y_test = labels

model = keras.models.load_model('final_model_lstm')
score, acc = model.evaluate(x_test, y_test,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)



