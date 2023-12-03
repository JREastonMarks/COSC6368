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
        while input_layer > (masking_size**2):
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
        if (layer < 0) or (actual == goal):
            return
        
        neuron_layer = self.layers[layer]
        neuron_layer_excitation_index = np.where(neuron_layer == 1)[0]

        weight = self.weights[layer]
        weight_calculated = np.multiply(weight, self.distances[layer])
        weight_calculated_neuron = weight_calculated[..., index]

        base_change = self.omega / (weight_calculated.shape[1] - 1)

        weight_change_index = -1
        weight_calculated_neuron_sorted_index = np.flip(np.argsort(weight_calculated_neuron))

        
        if actual < goal and (len(neuron_layer_excitation_index) == 0):
            # Scenario A - Actual 0, Goal 1, Excited == 0
            # Find the weight that would have the greatest impact. Use that weight to determine
            # which input neuron would be most likely to excite it. Then recursively call this function
            # to try to change it to 1
            weight_change_index = weight_calculated_neuron_sorted_index[len(weight_calculated_neuron_sorted_index) - 1]
            self._iterate_backward_propagation(0, 1, layer - 1, weight_change_index)
        elif actual < goal and (len(neuron_layer_excitation_index) > 0):
            # Scenario B - Actual 0, Goal 1, Excited > 0
            # Using the intersection of the index of sorted weights and of excited neurons 
            # determine the weight that is most likely to change the value of this neuron.
            # Increase that weight and decrease all other weights only if it is below 1 - omega.
            # Otherwise move to next one.
            valid_weight_index = self.__intersect_with_order(weight_calculated_neuron_sorted_index, neuron_layer_excitation_index)

            for candidate_weight_index in valid_weight_index:
                if weight[int(candidate_weight_index), index] < (1 - self.omega):
                    weight_change_index = int(candidate_weight_index)
                    break

            if weight_change_index >= 0:
                change = weight[weight_change_index]
                change = change - base_change
                change[index] = change[index] + self.omega + base_change

                weight[weight_change_index] = change

        else:
            # Scenario D - Actual 1, Goal 0, Excited > 0
            # Using the intersection of the index of sorted weights and of excited neurons
            # determine the weight that is most likely to change the value of the neuron.
            # Decrease that weight and increase all other weights only if it is above omega.
            # Otherwise move to next one.
            valid_weight_index = self.__intersect_with_order(weight_calculated_neuron_sorted_index, neuron_layer_excitation_index)

            for candidate_weight_index in valid_weight_index:
                if weight[int(candidate_weight_index), index] > self.omega:
                    weight_change_index = int(candidate_weight_index)
                    break

            if weight_change_index >= 0:
                change = weight[weight_change_index]
                change = change + base_change
                change[index] = change[index] - self.omega - base_change

                weight[weight_change_index] = change

    def __intersect_with_order(self, sorted_ar1, ar2):
        returns = np.empty([0,])

        for ar1_entry in sorted_ar1:
            if ar1_entry in ar2:
                returns = np.append(returns, ar1_entry)

        return returns
            


    # 4 Scenarios
    # A - Actual 0, Goal 1, Excited 0
    # B - Actual 0, Goal 1, Excited >0
    # C - Actual 1, Goal 0, Excited 0 - Cannot Happen
    # D - Actual 1, Goal 0, Excited >0
    def _iterate_backward_propagation_old(self, actual, goal, layer, index):
        if (layer < 0) or (actual == goal):
            return
        
        neuron_layer = self.layers[layer]
        neuron_layer_excitation_index = np.where(neuron_layer == 1)[0]

        weight = self.weights[layer]
        calc_weight = np.multiply(weight, self.distances[layer])
        
        base_change = self.omega / (calc_weight.shape[1] - 1)
        
        review_index = 0
        sorted_review_index = np.argsort(calc_weight[..., index])
        
        if actual < goal:
            # Scenario A and B
            if len(neuron_layer_excitation_index) == 0:
                # Scenario A
                review_index = np.argmax(calc_weight[..., index])
                self._iterate_backward_propagation(0, 1, layer - 1, review_index)
            else:
                # Scenario B
                sorted_review_index = np.flip(sorted_review_index)

                possible_weight_indexes = np.argwhere(weight[..., index] < (1 - self.omega)).flatten()
                
                for sorted_review_pos in sorted_review_index:
                    if ((sorted_review_pos in neuron_layer_excitation_index) and (sorted_review_pos in possible_weight_indexes)):
                        review_index = sorted_review_pos
                        break
                change = weight[review_index]  

                change = change - base_change
                change[index] = change[index] + self.omega + base_change

                weight[review_index] = change
        else:
            # Scenario D
            # sorted_review_index = np.flip(sorted_review_index)
            possible_weight_indexes = np.argwhere(weight[..., index] > self.omega).flatten()

            for sorted_review_pos in sorted_review_index:
                if ((sorted_review_pos in neuron_layer_excitation_index) and (sorted_review_pos in possible_weight_indexes)):
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