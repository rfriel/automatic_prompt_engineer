import random


def subsample_data(data, subsample_size, seed=None):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    if len(data) == 3:
        inputs, outputs, neg_outputs = data
    else:
        inputs, outputs = data
        neg_outputs = None
    assert len(inputs) == len(outputs)
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    if neg_outputs:
        neg_outputs = [neg_outputs[i] for i in indices]
        return inputs, outputs, neg_outputs
    return inputs, outputs


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
