import math

from einops import rearrange
import numpy as np
from tqdm import tqdm

from .components import BaseFitness, BaseMutation, BaseCrossing, BaseSelection


class GeneticOptimizer:

    def __init__(
            self,
            fitness: BaseFitness,
            crossing: BaseCrossing,
            mutation: BaseMutation,
            selection: BaseSelection,

            cross_prob: float = 0.7,
            mutate_prob: float = 0.1,

            n_chromosomes: int = 100,
            n_generations: int = 1000,
            n_elite: int = 0,
            patience: int = math.inf,
    ):
        self.fitness = fitness
        self.crossing = crossing
        self.mutation = mutation
        self.selection = selection

        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob

        self.n_chromosomes = n_chromosomes
        self.n_generations = n_generations
        self.n_elite = n_elite
        self.patience = patience

        self.genotypes = None
        self.fitnesses = None

        self.network = None
        self.max_params = None
        self.inputs = None
        self.outputs = None

    def optimize(self, network, inputs, outputs):
        self.network = network
        self.max_params = network.get_max_params()
        self.inputs = inputs
        self.outputs = outputs

        print(f'Optimizing {len(self.max_params)} parameters')

        self._generate_population()
        self._update_fitnesses()
        self._order_by_fitness()

        # -------early-stopping-handling-------- #
        best_fitness = -math.inf
        generations_without_change = 0
        # -------early-stopping-handling-------- #

        iterator = tqdm(range(self.n_generations), total=self.n_generations, desc='Evolution')
        for _ in iterator:
            self._evolve()
            accuracy = self.network.evaluate(self.inputs, self.outputs)
            iterator.set_postfix_str(f'fitness: {self.fitnesses[0]}, accuracy: {accuracy:.3f}')

            # -------early-stopping-handling-------- #
            if self.fitnesses[0] == math.inf:
                break
            if self.fitnesses[0] > best_fitness:
                best_fitness = self.fitnesses[0]
                generations_without_change = 0
            else:
                generations_without_change += 1
            if generations_without_change >= self.patience:
                break
            # -------early-stopping-handling-------- #

        self.network.set_params(self.genotypes[0])

    def _generate_population(self):
        random = np.random.randint(low=0, high=1_000_000, size=(self.n_chromosomes, len(self.max_params)))
        population = random % (self.max_params+1)
        self.genotypes = population

    def _update_fitnesses(self):
        fitnesses = []
        for genotype in self.genotypes:
            self.network.set_params(genotype)
            fitness = self.fitness(self.network, self.inputs, self.outputs)
            fitnesses.append(fitness)
        self.fitnesses = np.array(fitnesses)

    def _order_by_fitness(self):
        sorted_indexes = np.argsort(self.fitnesses)[::-1]
        self.genotypes = self.genotypes[sorted_indexes]
        self.fitnesses = self.fitnesses[sorted_indexes]

    def _evolve(self):
        elite = np.array(self.genotypes[:self.n_elite])
        selected = self._select(self.genotypes)
        crossed = self._cross(selected)
        mutated = self._mutate(crossed)

        self.genotypes = np.vstack([elite, mutated])
        self._update_fitnesses()
        self._order_by_fitness()

    def _select(self, chromosomes):
        return self.selection(chromosomes, self.fitnesses, self.n_chromosomes - self.n_elite)

    def _cross(self, genotypes):
        children = []
        np.random.shuffle(genotypes)
        paired_genotypes = rearrange(genotypes, '(n p) g -> n p g', p=2)
        for parent1, parent2 in paired_genotypes:
            children.extend(self.crossing(parent1, parent2))
        return np.array(children)

    def _mutate(self, genotypes):
        mutated = [self.mutation(genotype, self.max_params) for genotype in genotypes]
        return np.array(mutated)
