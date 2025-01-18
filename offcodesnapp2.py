import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Dataset
dataset = [
    "L7HZMNG6", "L7HY5U9C", "L7HXRBUT", "L7HU2PVL", "L7HSWZJZ",
    "L7HSC6PI", "L7HP9XNY", "L7HNCVJE", "L7HVCQLI", "L7HLWIMF",
    "L7HKU2J9", "L7HQ6SQD", "L7HPZ83L", "L7HI84AA", "L7HB7TBH",
    "L7H8952V", "L7H5WMHF", "L7H3C6DZ", "L7HCY69B"
]

# Split dataset: First 25% as features, last 75% as target
split_ratio = 0.25
features = [s[:int(len(s) * split_ratio)] for s in dataset]  # First 25%
targets = [s[int(len(s) * split_ratio):] for s in dataset]  # Last 75%

# ASCII encode both features and targets
max_feature_len = max(len(f) for f in features)
max_target_len = max(len(t) for t in targets)

encoded_features = np.array([
    [ord(char) for char in f.ljust(max_feature_len)] for f in features
])
encoded_targets = np.array([
    [ord(char) for char in t.ljust(max_target_len)] for t in targets
])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    encoded_features, encoded_targets, test_size=0.2, random_state=42
)

# Define the neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer (size = number of features)
    Dense(16, activation='relu'),     # Hidden layer
    Dense(y_train.shape[1])           # Output layer (size = target length)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model
evaluation = model.evaluate(X_test, y_test, verbose=1)

# Predict on the test set
predictions = model.predict(X_test)

# Decode predictions into characters
decoded_predictions = [
    ''.join(chr(int(round(value))) for value in pred) for pred in predictions
]

# Display results
print("\nEvaluation (Loss, MAE):", evaluation)
print("\nPredictions:")
for i, (feature, target, pred) in enumerate(zip(X_test, y_test, decoded_predictions)):
    feature_str = ''.join(chr(int(value)) for value in feature if value > 0)
    target_str = ''.join(chr(int(value)) for value in target if value > 0)
    print(f"Feature: {feature_str}, Target: {target_str}, Predicted: {pred}")

