import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn
from models.Hopfield import Hopfield

# Setup
label_encoder_b = CorticalMiniColumn(layer_sizes=[4,8,4])
label_somatic = Hopfield(size=4)
# label = np.array([1, 1, -1, -1], np.float32).reshape((1, 4))
label = np.array([1, 1, -1, -1], np.float32).reshape(4)

# Hopfield Test
query_a = label_somatic.query(label, 10)
label_somatic.train(label)
query_b = label_somatic.query(label, 10)
print(query_a)
print(query_b)

# Cortical Mini Column Test
print(label)

label_a = label_encoder_b.forward_propagation(label)
label_encoder_b.backward_propagation(False)

print(label_a)

