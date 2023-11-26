import numpy as np

def relu(Z):
    return np.sign(Z)

categoryA = np.array([[1, -1], [-1, 1]], np.float32)
categoryB = np.array([[1, -1], [1, -1]], np.float32)

inputA = categoryA.flatten();
inputB = categoryB.flatten();

weightsA = np.random.dirichlet(np.ones(24),size=1)
weightsA.resize(4,6)
weightsB = np.random.dirichlet(np.ones(24),size=1)
weightsB.resize(6,4)

# Middle Layer
layer1A = np.dot(inputA, weightsA)
layer1A = relu(layer1A)

layer1B = np.dot(inputB, weightsA)
layer1B = relu(layer1B)

# End Layer
layer2A = np.dot(layer1A, weightsB)
layer2A = relu(layer2A)
layer2B = np.dot(layer1B, weightsB)
layer2B = relu(layer2B)

print(f"{inputA} -> {layer1A} -> {layer2A}")
print(f"{inputB} -> {layer1B} -> {layer2B}")


