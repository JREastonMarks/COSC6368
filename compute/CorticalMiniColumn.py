import numpy as np

class CorticalMiniColumn:
    def __init__(self, layer_sizes=[4,6,4], omega=0.01, excitation=0.5 ) -> None:
        self.layer_sizes = layer_sizes
        self.omega = omega
        self.excitation = excitation
        self.weights = {}
        self.layers = {}
        
        

        for index in range(0,len(layer_sizes) - 1):
            first = layer_sizes[index]
            second = layer_sizes[index + 1]

            weight = np.empty([first, second], np.float16)
            for row in range(first):
                weight[row] = np.random.default_rng().dirichlet(np.ones(second), size=1)
            self.weights[index] = weight
            
    def forward_propagation(self, compute_array):
        self.layers[0] = compute_array
        for index in range(0,len(self.layer_sizes) - 1):
            self.layers[index + 1] = np.dot(self.layers[index], self.weights[index])
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