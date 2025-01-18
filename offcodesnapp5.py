import random
from collections import defaultdict

# Given dataset
data = [
    "L7HZMNG6", "L7HY5U9C", "L7HXRBUT", "L7HU2PVL", "L7HSWZJZ",
    "L7HSC6PI", "L7HP9XNY", "L7HNCVJE", "L7HVCQLI", "L7HLWIMF",
    "L7HKU2J9", "L7HQ6SQD", "L7HPZ83L", "L7HI84AA", "L7HB7TBH",
    "L7H8952V", "L7H5WMHF", "L7H3C6DZ", "L7HCY69B"
]

# Function to train a Markov Chain model
def train_markov_chain(data):
    transitions = defaultdict(lambda: defaultdict(int))
    for sequence in data:
        for i in range(len(sequence) - 1):
            curr_char = sequence[i]
            next_char = sequence[i + 1]
            transitions[curr_char][next_char] += 1
    return transitions

# Normalize transitions to probabilities
def normalize(transitions):
    normalized = {}
    for char, next_chars in transitions.items():
        total = sum(next_chars.values())
        normalized[char] = {k: v / total for k, v in next_chars.items()}
    return normalized

# Generate random alphanumeric string based on the Markov Chain
def generate_string(transitions, start_char='L', length=8):
    result = [start_char]
    for _ in range(length - 1):
        current_char = result[-1]
        if current_char not in transitions:
            break
        next_chars = list(transitions[current_char].keys())
        probabilities = list(transitions[current_char].values())
        next_char = random.choices(next_chars, probabilities)[0]
        result.append(next_char)
    return ''.join(result)

# Train the Markov Chain
transitions = train_markov_chain(data)
normalized_transitions = normalize(transitions)

# Generate random strings
random_strings = [generate_string(normalized_transitions) for _ in range(10)]
print("Generated Random Strings:")
print("\n".join(random_strings))

