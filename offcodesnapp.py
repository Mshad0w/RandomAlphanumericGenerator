import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Dataset
data = [
    "L7HZMNG6", "L7HY5U9C", "L7HXRBUT", "L7HU2PVL", "L7HSWZJZ",
    "L7HSC6PI", "L7HP9XNY", "L7HNCVJE", "L7HVCQLI", "L7HLWIMF",
    "L7HKU2J9", "L7HQ6SQD", "L7HPZ83L", "L7HI84AA", "L7HB7TBH",
    "L7H8952V", "L7H5WMHF", "L7H3C6DZ", "L7HCY69B"
]

# Character vocabulary
chars = sorted(list(set("".join(data))))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Convert data to integer sequences
sequences = [[char_to_idx[c] for c in seq] for seq in data]

# Prepare input and output pairs
X = []
y = []
for seq in sequences:
    for i in range(len(seq) - 1):
        X.append(seq[:i + 1])
        y.append(seq[i + 1])

# Pad sequences for equal length
max_len = max(len(x) for x in X)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='pre')
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))

# Model
model = Sequential([
    Embedding(input_dim=len(chars), output_dim=8, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X, y, epochs=200, batch_size=16)

# Generate new sequence
def generate_sequence(seed, length=8):
    generated = seed
    for _ in range(length - len(seed)):
        seq = [char_to_idx[c] for c in generated]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_len, padding='pre')
        next_char_idx = np.argmax(model.predict(seq, verbose=0))
        next_char = idx_to_char[next_char_idx]
        generated += next_char
    return generated

# Generate an example
print(generate_sequence("L7H"))

