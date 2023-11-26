import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn

# Setup

label_encoder = CorticalMiniColumn(layer_sizes=[9,4,2], omega=0.1)

label_input = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0], np.float32).reshape((9))
label_goal = np.array([1, 0], np.float32).reshape((2))

#Train 
for x in range(100):
    label_output = label_encoder.forward_propagation(label_input)

    print(f"{x}:\t{label_input} -> {label_output}")
    if(np.equal(label_goal, label_output).all()):
        break
    label_encoder.backward_propagation(label_goal)

print('fin')

