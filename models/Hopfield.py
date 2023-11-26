import numpy as np
from types import NoneType

class Hopfield:
    def __init__(self, size = -1, bias = 0, zero_base=False) -> None:
        self.size = size
        self.bias = bias
        self.zero_base = zero_base
        self.threshold = 0
        self.weights = None
        self.initialized = False


    def train(self, pattern):
        if self.zero_base:
            pattern[pattern == 0] = -1

        if(len(pattern.shape) == 1):
            pattern = pattern.reshape(1, pattern.shape[0])
        new_pattern = pattern * np.transpose(pattern)

        if self.initialized:
           self.weights = self.weights + new_pattern
        else:
            self.weights = new_pattern
            self.initialized = True
        
        if self.zero_base:
             pattern[pattern == -1] = 0

    def query(self, query, max_iterations) -> None:
        if self.initialized == False:
            return query
        
        if self.zero_base:
             query[query == 0] = -1
        
        energy = self.energy_function(query)

        for iteration in range(max_iterations):
            query = np.sign(query @ self.weights - self.threshold)

            new_energy = self.energy_function(query)

            if(new_energy == energy):
                 break
            energy = new_energy
        
        if self.zero_base:
             query[query == -1] = 0
        return query
                  

    def energy_function(self, query):
        query_transpose = query.transpose()
        part01 = query @ self.weights
        part02 = part01 @ query_transpose
        part03 = part02 * -.5
        return part03