import pandas as pd
import numpy as np
import pylab as plt

results = pd.read_csv('./model_history_log17.csv', delimiter = ',')
epoch = results['epoch'].tolist()
f1_train = results['get_f1'].tolist()
train_loss = results['loss'].tolist()
f1_val = results['val_get_f1'].tolist()
val_loss = results['val_loss'].tolist()

font = {'family' : 'normal',
        'size'   : 14}

fig = plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.plot(epoch,train_loss, label='train loss')
plt.plot(epoch,val_loss, label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Categorical Cross Entropy')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(epoch,f1_train, label='train F1')
plt.plot(epoch,f1_val, label='val F1')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend(loc='lower right')
plt.savefig('loss.png', bbox_inches = 'tight', pad_inches = 0)
