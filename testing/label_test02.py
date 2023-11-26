import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn
from models.Hopfield import Hopfield

vector_size = 4
# Setup
label_encoder = CorticalMiniColumn(layer_sizes=[vector_size,8,vector_size], omega=0.01)
label_decoder = CorticalMiniColumn(layer_sizes=[vector_size,8,vector_size], omega=0.01)

label_somatic = Hopfield(size=vector_size, zero_base=True)

label = np.array([1, 1, 0, 0], np.float32).reshape((vector_size))


label_vector = label_encoder.forward_propagation(label)
somatic_marker = label_somatic.query(label_vector, max_iterations=8)

if(np.equal(label_vector, somatic_marker).all()):
    label_somatic.train(label_vector)

    for x in range(1000):
        label_out = label_decoder.forward_propagation(somatic_marker)

        print(f"{x}\t{label} -> {label_vector} -> {label_out}")
        if(np.equal(label, label_out).all()):
            print(f"\tmatch {x}")
        else:
            label_decoder.backward_propagation(label)

print('fin')
