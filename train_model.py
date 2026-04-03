import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# Load dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake['label'] = 0
real['label'] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Padding
X_pad = pad_sequences(X_seq, maxlen=200)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# Model
model = Sequential()
model.add(Embedding(5000, 128, input_length=200))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Save model
model.save("model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved!")