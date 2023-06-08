import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# Load the emotion dataset
dataset_path = 'finalGPDATASET (3).csv'
data = pd.read_csv(dataset_path)

# Split the dataset into text and labels
texts = data['text'].values
labels = data['label'].values

# Preprocess the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Split the dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the model
embedding_dim = 100
hidden_units = 64
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    LSTM(hidden_units),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 10

# Train the model
model.fit(train_texts, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_texts, test_labels))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_texts, test_labels)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# # Make predictions on new samples
# new_samples = ['i am feeling eager to start doing some work the man who works there literally says so uhm you guys want to go in back and see if we can find anything to do']
# new_sequences = tokenizer.texts_to_sequences(new_samples)
# new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
# predicted_labels = model.predict(new_padded_sequences)
# predicted_emotions = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
#
# for sample, emotion in zip(new_samples, predicted_emotions):
#     print(f'Text: {sample}  Predicted Emotion: {emotion}')


pickle.dump(model,open("model.pkl","wb"))