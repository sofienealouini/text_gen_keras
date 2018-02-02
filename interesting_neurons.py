import numpy as np


# Load activations dictionary
dico = np.load("example/activations.npy").item()


# Exploration params
layer = 1
n = 10
threshold = 0.47


# Different selection scores
standard_devs = np.array([np.std(dico[layer][i]) for i in range(len(dico[layer]))])
max_mins = np.array([np.max(dico[layer][i] - np.min(dico[layer][i])) for i in range(len(dico[layer]))])
extreme_count = np.array([sum(np.abs(act) > threshold for act in neuron) for neuron in dico[layer]])


# Chosen score
score = extreme_count


# Get interesting neurons
indices = np.argpartition(score, -n)[-n:]
print(indices)
print(score[indices])