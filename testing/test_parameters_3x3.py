import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn

# Setup
vector_size = 3
label_input = np.array([0, 0, 1], np.float32).reshape((vector_size))
label_goal = np.array([1, 0, 0], np.float32).reshape((vector_size))

results = open("results_3x3.tsv", "x")

for omega_size in range(1, 90, 1):
    omega_size = omega_size / 100
    for excitation_level in range(1, 9, 1):
        excitation_level = excitation_level / 10
        

        solved = False
        sum_steps = 0
        #Train 
        for test_number in range(100):
            label_encoder = CorticalMiniColumn(layer_sizes=[vector_size,9,vector_size], omega=omega_size, excitation=excitation_level)

            for x in range(1000):    
                label_output = label_encoder.forward_propagation(label_input)

                if(np.equal(label_goal, label_output).all()):                    
                    solved = True
                    break
                label_encoder.backward_propagation(label_goal)
            sum_steps = sum_steps + x
        
        average = sum_steps / 100
        results.write(f"{omega_size}\t{excitation_level}\t{solved}\t{average}\n")
        results.flush()

print('fin')

