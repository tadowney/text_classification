import os
import numpy as np
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision, Recall
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import CSVLogger, Callback
from keras.optimizers import Adam

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch >= 8:  # or save after some epoch, each k-th epoch etc.
            self.model.save("model17_{}.hd5".format(epoch))

modelNum = 17
dataDir = './data/'
max_length = 400
padding = 'post'
embedding_dim = 300
ff_size = 128
num_classes = 13

# Load training data
f = open(os.path.join(dataDir, 'train.txt'), 'r', encoding="utf8")
trainSamples = f.readlines()
f.close()

t = Tokenizer()
t.fit_on_texts(trainSamples)
vocab_size = len(t.word_index) + 1

# integer encode the documents
train_sequences = t.texts_to_sequences(trainSamples)

f = open(os.path.join(dataDir, 'train.labels'), 'r', encoding="utf8")
trainLabels = f.readlines()
train_labels = np.asarray(trainLabels, dtype='int')
train_labels = to_categorical(train_labels)
#print(train_labels)
print(train_labels.shape)
f.close()

# Load validation data
f = open(os.path.join(dataDir, 'val.txt'), 'r', encoding="utf8")
valSamples = f.readlines()
f.close()

f = open(os.path.join(dataDir, 'val.labels'), 'r', encoding="utf8")
valLabels = f.readlines()
val_labels = np.asarray(valLabels, dtype='int')
val_labels = to_categorical(val_labels)
print(val_labels.shape)
f.close()


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(trainSamples)
vocab_size = len(t.word_index) + 1

# integer encode the documents
train_sequences = t.texts_to_sequences(trainSamples)
val_sequences = t.texts_to_sequences(valSamples)

# pad the documents to max_length size
train_pad = pad_sequences(train_sequences, maxlen=max_length, padding=padding)
val_pad = pad_sequences(val_sequences, maxlen=max_length, padding=padding)

# Load pretrained word embedding and create embedding matrix
embeddings_index = dict()
f = open('./glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# define model
model = Sequential()
e = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dropout(0.2))
model.add(Dense(ff_size, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

checkpoint = ModelCheckpoint("best_model" + str(modelNum) + ".hdf5", verbose=1,
    save_best_only=True, mode='auto', period=1)

csv_logger = CSVLogger("model_history_log" + str(modelNum) + ".csv", append=True)
saver = CustomSaver()

opt = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[get_f1])
num_epochs = 50
history = model.fit(train_pad, train_labels, epochs=num_epochs, validation_data=(val_pad, val_labels), callbacks=[checkpoint, saver, csv_logger], verbose=2)
