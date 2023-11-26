import numpy as np

class Network:
    def __init__(self, layer_sizes=[4,6,4], omega=0.01) -> None:
        self.layer_sizes = layer_sizes
        self.omega = omega
        self.weights = {}
        self.layers = {}
        

        for index in range(0,len(layer_sizes) - 1):
            first = layer_sizes[index]
            second = layer_sizes[index + 1]
            self.weights[index] = np.random.dirichlet(np.ones(first * second), size=1)
            self.weights[index].resize(first, second)
            
    def forward_propagation(self, input):
        next = input
        for index in range(0,len(self.layer_sizes) - 1):
            self.layers[index] = np.dot(next, self.weights[index])
            self.layers[index] = self.relu(self.layers[index])
            next = self.layers[index]

        return next
    
    def backward_propagation(self):
        for index in range(len(self.layer_sizes) - 2, -1, -1):
            layer = self.layers[index]
            weight = self.weights[index]

            print(index)
            print(layer)
            print(weight)
            pos =  self.omega
            neg = self.omega
            layer_sum = np.sum(self.layers[index])
            adj = self.omega / (abs(layer_sum) + 1)

            if(layer_sum > 0):
                pos = adj
            if(layer_sum < 0):
                neg = adj
            
            for layer_index, layer_value in enumerate(layer):
                if(layer_value > 0):
                    weight[:, layer_index] = weight[:, layer_index] + pos
                elif(layer_value < 0):
                    weight[:, layer_index] = weight[:, layer_index] - neg

    def relu(self, Z):
        return np.sign(Z)