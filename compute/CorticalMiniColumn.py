import numpy as np
import math

class CorticalMiniColumn:
    def __init__(self, input_size, output_size, masking_size, omega=0.01, excitation=0.5 ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.masking_size = masking_size
        self.omega = omega
        self.excitation = excitation
        self.weights = {}
        self.distances = {}
        self.layers = {}
        
        self.layer_sizes = [input_size]

        # Starting from the first position keep iterating until the previous layer size is less than the square root of the masking size
        input_layer = input_size
        layer_index = 0
        while input_layer > math.sqrt(input_size):
            layer_mask = self.__create_masking(int(math.sqrt(input_layer)), masking_size)
            self.distances[layer_index] = layer_mask
            output_layer = layer_mask.shape[1]
            weight = np.empty([input_layer, output_layer], np.float16)
            for row in range(input_layer):
                weight[row] = np.random.default_rng().dirichlet(np.ones(output_layer), size=1)
            self.weights[layer_index] = weight

            self.layer_sizes.append(output_layer)
            input_layer = output_layer
            layer_index = layer_index + 1

        # Connect the previous_layer to the output layer with a simple ones distance matrix
        self.distances[layer_index] = np.ones((input_layer, output_size))

        weight = np.empty([input_layer, output_size], np.float16)
        for row in range(input_layer):
            weight[row] = np.random.default_rng().dirichlet(np.ones(output_size), size=1)
        self.weights[layer_index] = weight
        self.layer_sizes.append(output_size)

            
    def forward_propagation(self, compute_array):
        self.layers[0] = compute_array
        for index in range(0,len(self.layer_sizes) - 1):
            # TODO: ADD MULTIPLY WEIGHTS TIMES DISTANCES
            self.layers[index + 1] = np.dot(self.layers[index], np.multiply(self.weights[index], self.distances[index]))
            self.layers[index + 1] = self._relu(self.layers[index + 1])
            compute_array = self.layers[index + 1]

        return compute_array
    

    def backward_propagation(self, goal):
        neuron_layer_last_position = len(self.layers) - 1
        neuron_last_layer = self.layers[neuron_layer_last_position]

        for goal_index in range(len(goal)):
            actual = neuron_last_layer[goal_index]
            self._iterate_backward_propagation(actual, goal[goal_index], neuron_layer_last_position - 1, goal_index)


    # 4 Scenarios
    # A - Actual 0, Goal 1, Excited 0
    # B - Actual 0, Goal 1, Excited >0
    # C - Actual 1, Goal 0, Excited 0 - Cannot Happen
    # D - Actual 1, Goal 0, Excited >0
    def _iterate_backward_propagation(self, actual, goal, layer, index):
        if (layer < 0):
            return
        if(actual == goal):
            return
        
        neuron_layer = self.layers[layer]
        neuron_layer_excitation_index = np.where(neuron_layer == 1)[0]

        weight = self.weights[layer]
        base_change = self.omega / (weight.shape[1] - 1)

        
        if actual < goal:
            # Scenario A and B
            if len(neuron_layer_excitation_index) == 0:
                # Scenario A
                review_index = np.argmax(weight[..., index])

                change = weight[review_index]  

                change = change - base_change
                change[index] = change[index] + self.omega + base_change

                weight[review_index] = change

                self._iterate_backward_propagation(0, 1, layer - 1, review_index)
            else:
                # Scenario B
                review_index = 0
                sorted_review_index = np.flip(np.argsort(weight[..., index]))
                for sorted_review_pos in sorted_review_index:
                    if sorted_review_pos in neuron_layer_excitation_index:
                        review_index = sorted_review_pos
                        break
                
                change = weight[review_index]  

                change = change - base_change
                change[index] = change[index] + self.omega + base_change

                weight[review_index] = change

        else:
            # Scenario D
            review_index = 0
            sorted_review_index = np.argsort(weight[..., index])
            for sorted_review_pos in sorted_review_index:
                if sorted_review_pos in neuron_layer_excitation_index:
                    review_index = sorted_review_pos
                    break
            
            change = weight[review_index]  

            change = change + base_change
            change[index] = change[index] - self.omega - base_change

            weight[review_index] = change

    
    def _relu(self, Z):

        Z = np.clip(Z, 0, 1)
        Z[Z >=self.excitation] = 1
        Z[Z < self.excitation] = 0
        return Z
    
    def __create_mask(self, matrix_size, dither_size, dither_x, dither_y):
        mask = np.zeros((matrix_size, matrix_size))
        offset = math.floor(dither_size / 2)

        dither_x_start = dither_x - offset
        dither_y_start = dither_y - offset

        dither_x_end = dither_x + offset
        dither_y_end = dither_y + offset

        for pos_x in range(0, matrix_size):
            for pos_y in range(0, matrix_size):
                if (dither_x_start <= pos_x <= dither_x_end) and (dither_y_start <= pos_y <= dither_y_end):
                    weight = 1
                else:
                    weight = 0.5 # Change to an offset 2^-x
                mask[pos_x, pos_y] = weight
        
        return mask

    def __create_masking(self, matrix_size, dither_size):
        offset = math.floor(dither_size / 2)
        start = offset
        end = matrix_size - offset
        mask = np.zeros((matrix_size * matrix_size, 0))

        for dither_x in range(start, end):
            for dither_y in range(start, end):
                submask = self.__create_mask(matrix_size, dither_size, dither_x, dither_y)
                submask.resize(matrix_size * matrix_size, 1)
                
                mask = np.append(mask, submask, axis=1)

        return mask