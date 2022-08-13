'''import tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
from sklearn import preprocessing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

dataset=pd.read_csv('ecgvalues.csv')
X = dataset.drop('target' ,axis=1)
y = dataset.target
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.15)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential([
    Dense(32, activation='relu', input_shape=(15,)),
    Dense(64, activation='relu'),
    #Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

 
hist = model.fit(X_train, y_train,
          batch_size=32, epochs=100)

predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix

for i in range(len(predictions)):
  if(predictions[i]>0.5):
    predictions[i]=1
  else: predictions[i]=0
confusion_matrix(y_test, predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

model.save("trained_model")'''



import tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
from sklearn import preprocessing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


dataset=pd.read_csv('ecgvalues.csv')
X = dataset.drop('target' ,axis=1)
y = dataset.target
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.15)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

clf.fit(X_train,y_train)

predictions=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

# clf.save("trained_model")