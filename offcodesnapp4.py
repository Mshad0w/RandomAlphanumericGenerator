import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Dataset
data = [
    "L7HZMNG6", "L7HY5U9C", "L7HXRBUT", "L7HU2PVL",
    "L7HSWZJZ", "L7HSC6PI", "L7HP9XNY", "L7HNCVJE",
    "L7HVCQLI", "L7HLWIMF", "L7HKU2J9", "L7HQ6SQD",
    "L7HPZ83L", "L7HI84AA", "L7HB7TBH", "L7H8952V",
    "L7H5WMHF", "L7H3C6DZ", "L7HCY69B"
]

# Preprocess Data
chars = sorted(set("".join(data)))  # Unique characters in the dataset
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Create sequences for training
sequence_length = 5
sequences = []
next_chars = []

for code in data:
    for i in range(len(code) - sequence_length):
        seq = code[i:i+sequence_length]
        sequences.append([char_to_index[char] for char in seq])
        next_chars.append(char_to_index[code[i+sequence_length]])

sequences = np.array(sequences)
next_chars = np.array(next_chars)

# Build the Model
model = Sequential([
    Embedding(input_dim=len(chars), output_dim=50, input_length=sequence_length),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
model.fit(sequences, next_chars, epochs=50, batch_size=32)

# Generate Data
def generate_data(seed, length=8):
    result = seed
    for _ in range(length - len(seed)):
        x = np.array([[char_to_index[char] for char in result[-sequence_length:]]])
        preds = model.predict(x, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = index_to_char[next_index]
        result += next_char
    return result

# Generate samples
seed = "L7H"
for _ in range(5):
    print(generate_data(seed))

