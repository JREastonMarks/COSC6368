import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn

# Setup
vector_size = 2

label_encoder = CorticalMiniColumn(layer_sizes=[vector_size,4,vector_size], omega=0.1)

label_input = np.array([0, 1], np.float32).reshape((vector_size))
label_goal = np.array([1, 0], np.float32).reshape((vector_size))

#Train 
for x in range(10):
    label_output = label_encoder.forward_propagation(label_input)

    print(f"{x}:\t{label_input} -> {label_output}")
    if(np.equal(label_goal, label_output).all()):
        break
    label_encoder.backward_propagation(label_goal)

print('fin')

