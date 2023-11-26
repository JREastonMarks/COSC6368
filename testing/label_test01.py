import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn
from models.Hopfield import Hopfield

vector_size = 4
# Setup
label_encoder = CorticalMiniColumn(layer_sizes=[vector_size,16,vector_size], omega=0.01)
label_decoder = CorticalMiniColumn(layer_sizes=[vector_size,16,vector_size], omega=0.01)

label_somatic = Hopfield(size=vector_size)

label = np.array([1, 1, -1, -1], np.float32).reshape((vector_size))

for x in range(1000):
    
    label_vector = label_encoder.forward_propagation(label)
    somatic_marker = label_somatic.query(label_vector, max_iterations=8)
    label_out = label_decoder.forward_propagation(somatic_marker)

    print(f"{x}\t{label} -> {label_vector} -> {label_out}")
    if(np.equal(label, label_out).all()):
        print(f"\tmatch {x}")
        label_encoder.backward_propagation(True)
        label_decoder.backward_propagation(True)
        label_somatic.train(label_vector)
        break
    else:
        # print("\tnope")
        if x % 2 == 1:
            label_encoder.backward_propagation(False)
        else:
            label_decoder.backward_propagation(False)

print('fin')
