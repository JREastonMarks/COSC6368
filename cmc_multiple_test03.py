import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn

# Setup
vector_input_size = 25
vector_output_size = 3

label_encoder = CorticalMiniColumn(layer_sizes=[vector_input_size, 100, vector_output_size], omega=0.01, excitation=0.6)

label_a_input = np.eye(5, dtype=np.float32).reshape((vector_input_size))
label_a_goal = np.array([1, 0, 0], np.float32).reshape((vector_output_size))

label_b_input = np.flip(np.eye(5, dtype=np.float32), axis=1).reshape((vector_input_size))
label_b_goal = np.array([0, 1, 0], np.float32).reshape((vector_output_size))

# label_c_input = np.array([1, 0, 0, 1,
                        #   0, 1, 1, 0,
                        #   0, 1, 1, 0,
                        #   1, 0, 0, 1], np.float32).reshape((vector_input_size))
label_c_input = np.add(label_a_input, label_b_input)
label_c_input[label_c_input > 1] = 1
label_c_goal = np.array([0, 0, 1], np.float32).reshape((vector_output_size))

def train(unit, label_in, label_goal):
    for x in range(10000):
        label_output = label_encoder.forward_propagation(label_in)

        # print(f"{unit}\t{x}:\t{label_in} -> {label_output}")
        if(np.equal(label_output, label_goal).all()):
            break
        label_encoder.backward_propagation(label_goal)


#Train A
train("a", label_a_input, label_a_goal)

#Train B
train("b", label_b_input, label_b_goal)

#Train C
train("c", label_c_input, label_c_goal)

#Test

label_a_output = label_encoder.forward_propagation(label_a_input)
print(f"a:\t{label_a_output}")

label_b_output = label_encoder.forward_propagation(label_b_input)
print(f"b:\t{label_b_output}")

label_c_output = label_encoder.forward_propagation(label_c_input)
print(f"c:\t{label_c_output}")

print("fin")