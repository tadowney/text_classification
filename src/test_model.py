import os
import numpy as np
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model
from sklearn.metrics import f1_score

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

calcValF1 = True

dataDir = './data/'
modelDir = './'
modelNum = 'model17_14.hd5'
max_length = 400
padding = 'post'

# Load training data
f = open(os.path.join(dataDir, 'train.txt'), 'r', encoding="utf8")
trainSamples = f.readlines()
f.close()

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(trainSamples)
vocab_size = len(t.word_index) + 1

print('Loading model {}'.format(modelNum))
model = load_model(os.path.join(modelDir, modelNum), custom_objects={"get_f1": get_f1})

if(calcValF1):
    print("Loading validation data...")

    # Load validation data
    f = open(os.path.join(dataDir, 'val.txt'), 'r', encoding="utf8")
    valSamples = f.readlines()
    f.close()

    f = open(os.path.join(dataDir, 'val.labels'), 'r', encoding="utf8")
    valLabels = f.readlines()
    y_true = np.asarray(valLabels, dtype='int')
    val_labels = to_categorical(y_true)
    print(val_labels.shape)
    f.close()

    val_sequences = t.texts_to_sequences(valSamples)
    val_pad = pad_sequences(val_sequences, maxlen=max_length, padding=padding)

    print('Predicting on validation samples...')
    pred = model.predict(val_pad)

    print('Calculating f1 score...')
    y_pred = np.zeros(len(y_true), dtype='int')
    for i,p in enumerate(pred):
        y_pred[i] = int(np.argmax(p))

    val_f1 = f1_score(y_true, y_pred, average='micro')
    print('Validation F1: {}'.format(val_f1))



