import numpy as np
from compute.Network import Network

network = Network(layer_sizes=[4,4,2])

categoryA = np.array([[1, -1], [-1, 1]], np.float32)
inputA = categoryA.flatten()

categoryB = np.array([[1, -1], [1, -1]], np.float32)
inputB = categoryB.flatten()


resultA = network.forward_propagation(inputA)
resultB = network.forward_propagation(inputB)

print(f"\nA:\t{inputA} -> {resultA}")
print(f"B:\t{inputB} -> {resultB}")

for x in range(100):
    resultA = network.forward_propagation(inputA)
    network.backward_propagation()

for x in range(100):
    resultB = network.forward_propagation(inputB)
    network.backward_propagation()


print(f"\nA:\t{inputA} -> {resultA}")
print(f"B:\t{inputB} -> {resultB}")