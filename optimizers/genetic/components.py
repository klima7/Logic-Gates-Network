from abc import ABC, abstractmethod

import numpy as np


class BaseCrossing(ABC):

    @abstractmethod
    def __call__(self, first_parent, second_parent):
        pass


class BaseFitness(ABC):

    @abstractmethod
    def __call__(self, network, inputs, outputs):
        pass


class BaseMutation(ABC):

    @abstractmethod
    def __call__(self, chromosome, max_params):
        pass


class BaseSelection(ABC):

    @abstractmethod
    def __call__(self, chromosomes, fitnesses, select_count):
        pass


class RouletteWheelSelection(BaseSelection):

    def __call__(self, chromosomes, fitnesses, select_count):
        all_indexes = np.arange(len(chromosomes))
        probability = fitnesses / np.sum(fitnesses)
        selected_indexes = np.random.choice(all_indexes, size=select_count, p=probability)
        return np.array(chromosomes[selected_indexes])


class RankSelection(BaseSelection):

    def __init__(self, count):
        self.count = count

    def __call__(self, chromosomes, fitnesses, select_count):
        new_chromosomes = chromosomes[:select_count]
        new_chromosomes[select_count-self.count:] = chromosomes[:self.count]
        return np.array(new_chromosomes)


class NRandomChangesMutation(BaseMutation):

    def __init__(self, n_changes):
        self.n_changes = n_changes

    def __call__(self, chromosome, max_params):
        indexes = np.arange(len(chromosome))
        np.random.shuffle(indexes)
        indexes = indexes[:self.n_changes]
        chromosome[indexes] = np.random.randint(low=0, size=(self.n_changes,)) % (max_params[indexes] + 1)
        return chromosome


class MultiPointCrossing(BaseCrossing):

    def __init__(self, n_points=1):
        self.n_points = n_points

    def __call__(self, first_parent, second_parent):
        split_indexes = np.arange(1, len(first_parent)-1)
        np.random.shuffle(split_indexes)
        split_indexes = np.sort(split_indexes[:self.n_points])

        merged_parents = np.row_stack([first_parent, second_parent])
        parts = np.split(merged_parents, split_indexes, axis=1)

        for part in parts:
            np.random.shuffle(part)

        reordered_merged = np.hstack(parts)
        first_child, second_child = reordered_merged[0], reordered_merged[1]
        return first_child, second_child


class SmallestMaeFitness(BaseFitness):

    def __call__(self, network, inputs, outputs):
        predictions = network.predict(inputs)
        error = np.sum(predictions ^ outputs) / len(inputs)
        fitness = 1 / error
        return fitness
