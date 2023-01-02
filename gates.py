import random

import numpy as np

import operations as ops


class Gate:

    OPERATIONS = [ops.or_, ops.and_, ops.nand, ops.nor, ops.xor]

    def __init__(self, input_size):
        self.input_size = input_size

        self.neg1 = random.randint(0, 1)
        self.neg2 = random.randint(0, 1)
        self.arg1 = random.randint(0, input_size-1)
        self.arg2 = random.randint(0, input_size-1)
        self.op = random.randint(0, len(self.OPERATIONS)-1)

    def predict(self, inputs):
        op = self.OPERATIONS[self.op]
        arg1 = inputs[self.arg1]
        arg2 = inputs[self.arg2]

        if self.neg1:
            arg1 = not arg1
        if self.neg2:
            arg1 = not arg2

        return op(arg1, arg2)

    def get_params(self):
        return [self.neg1, self.neg2, self.arg1, self.arg2, self.op]

    def set_params(self, params):
        assert len(params) == 5
        self.neg1, self.neg2, self.arg1, self.arg2, self.op = params

    def get_max_params(self):
        return [1, 1, self.input_size-1, self.input_size-1, len(self.OPERATIONS)-1]


class Layer:

    def __init__(self, size, input_size):
        self.size = size
        self.gates = [Gate(input_size) for _ in range(size)]

    def predict(self, inputs):
        predictions = [gate.predict(inputs) for gate in self.gates]
        return predictions

    def get_params(self):
        nested_params = np.array([gate.get_params() for gate in self.gates])
        params = nested_params.flatten()
        return params

    def set_params(self, params):
        nested_params = np.array(params).reshape(len(self.gates), 5)
        for gate, params in zip(self.gates, nested_params):
            gate.set_params(params)

    def get_max_params(self):
        nested_params = np.array([gate.get_max_params() for gate in self.gates])
        params = nested_params.flatten()
        return params


class Network:

    def __init__(self, input_size, hidden_layers_sizes):
        self.input_size = input_size
        self.layers = self.__create_layers(input_size, hidden_layers_sizes)

    def evaluate(self, inputs, outputs):
        predictions = self.predict(inputs)
        loss = self.__get_loss(predictions, outputs)
        accuracy = self.__get_accuracy(predictions, outputs)
        return loss, accuracy

    def predict(self, multiple_inputs):
        outputs = [self.predict_single(single_inputs) for single_inputs in multiple_inputs]
        return np.array(outputs)

    def predict_single(self, inputs):
        assert len(inputs) == self.input_size
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.predict(current_inputs)
        return current_inputs

    def get_params(self):
        nested_params = [layer.get_params() for layer in self.layers]
        params = [item for sublist in nested_params for item in sublist]
        return np.array(params)

    def set_params(self, params):
        remaining_params = list(params)
        for layer in self.layers:
            params = remaining_params[:layer.size*5]
            remaining_params = remaining_params[layer.size*5:]
            layer.set_params(params)

    def get_max_params(self):
        nested_params = [layer.get_max_params() for layer in self.layers]
        params = [item for sublist in nested_params for item in sublist]
        return np.array(params)

    @staticmethod
    def __create_layers(input_size, hidden_layers_sizes):
        layers = []
        all_sizes = [input_size, *hidden_layers_sizes]
        for input_size, gates_count in zip(all_sizes, all_sizes[1:]):
            layer = Layer(size=gates_count, input_size=input_size)
            layers.append(layer)
        return layers

    @staticmethod
    def __get_accuracy(predictions, targets):
        predictions, targets = predictions > 0.5, targets > 0.5
        correct_count = np.sum(np.all(predictions == targets, axis=1))
        total_count = len(targets)
        return correct_count / total_count if total_count != 0 else 0

    @staticmethod
    def __get_loss(predictions, targets):
        predictions, targets = predictions.astype(np.float64), targets.astype(np.float64)
        partial_losses = np.mean(np.power(targets - predictions, 2), axis=1)
        mean_loss = np.mean(partial_losses)
        return mean_loss
