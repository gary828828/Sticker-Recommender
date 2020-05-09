import numpy as np
import pandas as pd
import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Activation, Dropout, Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv('text_emotion_train_val_set.csv')
df = df.loc[df['sentiment'].isin(['happiness','sadness','surprise','hate','love'])]
df = df.drop(['tweet_id', 'author'], axis=1)

df = df.sample(frac=1).reset_index(drop=True)

y_pandas_df = pd.get_dummies(df['sentiment'])
y = y_pandas_df.values
x_train, x_test, y_train, y_test = train_test_split(df,y, random_state=4, test_size=0.2)

x_train = x_train.drop(['sentiment'], axis=1)
x_test = x_test.drop(['sentiment'], axis=1)

x_train = x_train.values.flatten()
x_test = x_test.values.flatten()

list_of_classes = ['happiness','sadness','surprise','hate','love']
max_features = 20000
max_text_length = 400
embedding_dims = 50
batch_size = 32
epochs = 3
num_filters_1 = 250
num_filters_2 = 250
filter_size = 3

x_tokenizer = text.Tokenizer(num_words=max_features)
x_tokenizer.fit_on_texts(list(x_train))
x_tokenized = x_tokenizer.texts_to_sequences(x_train)
x_train= sequence.pad_sequences(x_tokenized, maxlen=max_text_length)
#----------------------------model----------------------------------------------
model = Sequential()

model.add(Embedding(max_features, embedding_dims, input_length=max_text_length))
model.add(Dropout(0.2))

model.add(Conv1D(filters=num_filters_1, kernel_size=filter_size, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(num_filters_2))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

model.save('modelSR.h5')
# Loading model to compare the results
model = tf.keras.models.load_model('modelSR.h5')
