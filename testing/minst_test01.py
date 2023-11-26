import numpy as np
from keras.datasets import mnist 
from matplotlib import pyplot

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    image = train_X[i]

    image = (image-np.min(image))/(np.max(image)-np.min(image))

    image[image >= 0.5] = 1
    image[image < 0.5] = 0

    print(train_y[i])
    pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))

pyplot.show()