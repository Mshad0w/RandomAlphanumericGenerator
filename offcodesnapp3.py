# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Training dataset provided
dataset = [
    "L7HZMNG6", "L7HY5U9C", "L7HXRBUT", "L7HU2PVL", "L7HSWZJZ",
    "L7HSC6PI", "L7HP9XNY", "L7HNCVJE", "L7HVCQLI", "L7HLWIMF",
    "L7HKU2J9", "L7HQ6SQD", "L7HPZ83L", "L7HI84AA", "L7HB7TBH",
    "L7H8952V", "L7H5WMHF", "L7H3C6DZ", "L7HCY69B"
]

# Prepare a character-level dictionary for encoding
chars = sorted(set("".join(dataset)))  # Unique characters in the dataset
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for char, i in char_to_index.items()}

# Encode the dataset
encoded_dataset = [[char_to_index[char] for char in code] for code in dataset]

# Prepare data for training
sequence_length = max(len(code) for code in dataset)
X = []
y = []

for sequence in encoded_dataset:
    for i in range(1, len(sequence)):
        X.append(sequence[:i])
        y.append(sequence[i])

# Pad sequences and one-hot encode targets
X = pad_sequences(X, maxlen=sequence_length, padding='pre')
y = to_categorical(y, num_classes=len(chars))

# Build the model
model = Sequential([
    Embedding(input_dim=len(chars), output_dim=8, input_length=sequence_length),
    LSTM(64, return_sequences=False),
    Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)

# Function to generate synthetic sequences
def generate_sequence(seed_text, length=8):
    seed_encoded = [char_to_index[char] for char in seed_text]
    for _ in range(length - len(seed_text)):
        padded_seed = pad_sequences([seed_encoded], maxlen=sequence_length, padding='pre')
        prediction = model.predict(padded_seed, verbose=0)
        next_char_index = np.argmax(prediction)
        seed_encoded.append(next_char_index)
    return "".join(index_to_char[i] for i in seed_encoded)

# Generate synthetic codes
print("Generated Codes:")
for _ in range(10):
    print(generate_sequence("L7H"))

