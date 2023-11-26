import numpy as np
from compute.CorticalMiniColumn import CorticalMiniColumn

data_mini_column = CorticalMiniColumn(layer_sizes=[4,8,4])
label_mini_column = CorticalMiniColumn(layer_sizes=[4,8,4])

#CATEGORY A
category_A_data = np.array([[1, -1], [-1, 1]], np.float32)
input_A = category_A_data.flatten()
category_A_label = np.array([1,0,0,0])

category_A_data_result = data_mini_column.forward_propagation(input_A)
category_A_label_result = label_mini_column.forward_propagation(category_A_label)

category_A_Engram = np.concatenate((category_A_data_result, category_A_label_result))
print(category_A_Engram)


#CATEGORY B
category_B_Data = np.array([[1, -1], [1, -1]], np.float32)
input_B = category_B_Data.flatten()
category_B_label = np.array([0,1,0,0])

category_B_data_result = data_mini_column.forward_propagation(input_B)
category_B_label_result = label_mini_column.forward_propagation(category_B_label)

category_B_Engram = np.concatenate((category_B_data_result, category_B_label_result))
print(category_B_Engram)
