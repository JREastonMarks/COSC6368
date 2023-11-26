import numpy as np
from keras.datasets import mnist 
from compute.CorticalMiniColumn import CorticalMiniColumn

cmm = CorticalMiniColumn(layer_sizes=[784,784,10], omega=0.6, excitation=0.5)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

total = 0
failed = 0
passed = 0
for i in range(len(train_X)):  
    image = train_X[i]

    image = (image-np.min(image))/(np.max(image)-np.min(image))
    image[image >= 0.5] = 1
    image[image < 0.5] = 0
    image = image.flatten()

    goal_number = train_y[i]
    goal_vector = np.zeros(shape=(10))
    goal_vector[goal_number] = 1

    output = cmm.forward_propagation(image)

    print(f"{i}:\t{goal_number} -> {output}")
    total += 1
    if(not np.equal(goal_vector, output).all()):
        failed += 1
        cmm.backward_propagation(goal_vector)
    else:
        passed += 1

print(f"Correct: {passed}/{total}")
print(f" Failed: {failed}/{total}")