import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn


# Setup
vector_input_size = 25
vector_output_size = 3
dither_size = 3

label_encoder = CorticalMiniColumn(vector_input_size, vector_output_size, dither_size, omega=0.01, excitation=0.7)

label_a_input = np.array([1, 0, 0, 0, 0,
                          0, 1, 0, 0, 0,
                          0, 0, 1, 0, 0,
                          0, 0, 0, 1, 0,
                          0, 0, 0, 0, 1], np.float32).reshape((vector_input_size))

label_a_goal = np.array([1, 0, 0], np.float32).reshape((vector_output_size))

label_b_input = np.array([0, 0, 0, 0, 1, 
                          0, 0, 0, 1, 0,
                          0, 0, 1, 0, 0,
                          0, 1, 0, 0, 0,
                          1, 0, 0, 0, 0], np.float32).reshape((vector_input_size))
label_b_goal = np.array([0, 0, 1], np.float32).reshape((vector_output_size))

label_c_input = np.array([1, 0, 0, 0, 1, 
                          0, 1, 0, 1, 0,
                          0, 0, 1, 0, 0,
                          0, 1, 0, 1, 0,
                          1, 0, 0, 0, 1], np.float32).reshape((vector_input_size))
label_c_goal = np.array([1, 0, 1], np.float32).reshape((vector_output_size))

def train(unit, label_in, label_goal):
    result = -1
    for x in range(1000):
        label_output = label_encoder.forward_propagation(label_in)

        
        if(np.equal(label_output, label_goal).all()):
            result = x
            break
        label_encoder.backward_propagation(label_goal)
    return result

a_failed = 0
a_sum = 0
b_failed = 0
b_sum = 0
c_failed = 0
c_sum = 0
all_sum = 0

for experiment in range(1000):
    all_passed = True
    label_encoder = CorticalMiniColumn(vector_input_size, vector_output_size, dither_size, omega=0.01, excitation=0.7)
    #Train A
    a_trained = train("a", label_a_input, label_a_goal)

    #Train B
    b_trained = train("b", label_b_input, label_b_goal)

    #Train C
    c_trained = train("c", label_c_input, label_c_goal)


    label_a_output = label_encoder.forward_propagation(label_a_input)
    label_b_output = label_encoder.forward_propagation(label_b_input)
    label_c_output = label_encoder.forward_propagation(label_c_input)

    if(not np.equal(label_a_output, label_a_goal).all()):
        all_passed = False
        a_failed = a_failed + 1
    
    if(not np.equal(label_b_output, label_b_goal).all()):
        all_passed = False
        b_failed = b_failed + 1

    if(not np.equal(label_c_output, label_c_goal).all()):
        all_passed = False
        c_failed = c_failed + 1
    
    if all_passed:
        all_sum = all_sum + 1
        a_sum = a_sum + a_trained
        b_sum = b_sum + b_trained
        c_sum = c_sum + c_trained

print(f"All Sum: {all_sum}")
print(f"A sum: {a_sum}")
print(f"A failed: {a_failed}")
if all_sum > 0:
    print(f"A average: {a_sum / all_sum}")

print(f"B sum: {b_sum}")
print(f"B failed: {b_failed}")
if all_sum > 0:
    print(f"B average: {b_sum / all_sum}")

print(f"C sum: {c_sum}")
print(f"C failed: {c_failed}")
if all_sum > 0:
    print(f"C average: {c_sum / all_sum}")

print("fin")