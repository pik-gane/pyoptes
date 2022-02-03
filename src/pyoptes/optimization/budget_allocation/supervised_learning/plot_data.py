
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
def postprocessing(input_data, target_data, split):
    
    input_data = pd.read_csv(input_data, header = None, sep = ',')
    target_data = pd.read_csv(target_data, header = None, sep = ',')
    
    is_NaN = input_data.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = input_data[row_has_NaN]
    input_data = input_data.drop(rows_with_NaN)
    target_data = target_data.drop(rows_with_NaN)

    subset_training_input = input_data.iloc[split:]
    subset_training_targets = target_data.iloc[split:]

    subset_test_input = input_data.iloc[0:split]
    subset_test_targets = target_data.iloc[0:split]

    return subset_training_input, subset_training_targets, subset_test_input, subset_test_targets


#input_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/input_data_full.csv"
#target_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/targets_data_full.csv"

input_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/input_data_none.csv"
target_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/targets_data_none.csv"

train_input, train_targets, test_input, test_targets = postprocessing(input_data, target_data, split = 300)

print(f'\nSize of training set: {len(train_input)}, Size of test set: {len(test_input)}')
print(f'\n\nTrain inputs mean: {train_input.to_numpy().mean()}, Train inputs std: {train_input.to_numpy().std()}')
print(f'\n\nTrain targets mean: {train_targets.to_numpy().mean()}, Train targets std: {train_targets.to_numpy().std()}')
print(f'\n\nTest inputs mean: {test_input.to_numpy().mean()}, Test inputs std: {test_input.to_numpy().std()}')
print(f'\n\nTest targets mean: {test_targets.to_numpy().mean()}, Train targets std: {test_targets.to_numpy().std()}')


#plt.subplots()

plt.figure("Training inputs", figsize=(5,5))
for i in range(len(train_input)):
    plt.plot(np.arange(121), train_input.iloc[i])
    plt.axis([0, 121, 0, 130])
    plt.title("Training Inputs")
    #plt.xlabel="Nodes"
    #plt.ylabel="Budget"

plt.figure("Training targets", figsize=(5,5))
plt.plot(np.arange(len(train_targets)), train_targets)
plt.title("Training targets")
#plt.ylabel="Infected animals"
#plt.xlabel="Samples"

plt.figure("Test inputs", figsize=(5,5))
for i in range(len(test_input)):
    plt.plot(np.arange(121), test_input.iloc[i])
    plt.axis([0, 121, 0, 130])
    plt.title("Test inputs")

plt.figure("Test targets", figsize=(5,5))
plt.plot(np.arange(len(test_targets)), test_targets)
plt.title("Test targets")

plt.show()
