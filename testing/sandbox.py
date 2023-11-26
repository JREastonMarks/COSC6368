import numpy as np

input = np.array([0, 0, 1, 1], np.float32)

# matrix = np.random.dirichlet(np.ones(6), size = 1)
# matrix.resize(2,3)
# matrix = np.array([[1,2,4], [8,16,32]], np.float32)
matrix = np.array([[1, 3, 5, 7],[2, 4, 6, 8]], np.float32)

output = matrix.dot(input)

print(output)

# output = np.sign(output)
# print(input)
# print(matrix)
# print(output)
# print(np.sum(matrix))

# matrix2 = matrix
# matrix2[0] = matrix2[0] + .01
# matrix2[1] = matrix2[1] - .01


# output2 = matrix.dot(input)

# output2 = np.sign(output2)
# print(input)
# print(matrix2)
# print(output2)
# print(np.sum(matrix2))
