import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn

# Any live cell with fewer than two live neighbours dies, as if by underpopulation.
# Any live cell with two or three live neighbours lives on to the next generation.
# Any live cell with more than three live neighbours dies, as if by overpopulation.
# Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction

cell = CorticalMiniColumn(layer_sizes=[10,10,1], omega=0.1, excitation=0.5)

def train(unit, label_in, label_goal):
    for x in range(10000):
        label_output = label_encoder.forward_propagation(label_in)

        # print(f"{unit}\t{x}:\t{label_in} -> {label_output}")
        if(np.equal(label_output, label_goal).all()):
            break
        label_encoder.backward_propagation(label_goal)


test = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]], np.int8)

print(test)
print(test.flatten())

cell = np.array([0], np.int8)
print(cell)
cell = np.append(cell, test.flatten())

print(cell)