# START
# Generate the initial population
# Compute fitness
# REPEAT
#     Selection
#     Crossover
#     Mutation
#     Compute fitness
# UNTIL population has converged
# STOP
from typing import Set, List, Tuple

import numpy as np
import random
import itertools
from enum import Enum

Population = List[np.ndarray]
PopulationWithFitness = List[Tuple[float, np.ndarray]]


class SelectionAlgorithm(Enum):
    foo = 1


class FitnessFunction(Enum):
    AveragePrecision = 1
    ROCCurve = 2


def random_features_indices(indices_count: int):
    return random.sample(range(Algorithm.ALL_ATTRIBUTES_COUNT), indices_count)


class Algorithm:
    ALL_ATTRIBUTES_COUNT = 100_000

    def __init__(self, classifier, *, population_size=10, min_attributes_count=10,
                 initial_attributes_standard_deviation=1, individuals_to_mutate_coefficient=0.3,
                 chromosomes_to_mutate_coefficient=0.05, cycles_count=4, loci_count=2,
                 fitness_function=FitnessFunction.AveragePrecision, selection_algorithm=SelectionAlgorithm.foo):
        self.selection_algorithm = selection_algorithm
        self.fitness_function = fitness_function
        self.loci_count = loci_count if loci_count % 2 == 0 else loci_count + 1
        self.cycles_count = cycles_count
        self.chromosomes_to_mutate_coefficient = chromosomes_to_mutate_coefficient
        self.individuals_to_mutate_coefficient = individuals_to_mutate_coefficient
        self.initial_attributes_standard_deviation = initial_attributes_standard_deviation
        self.min_attributes_count = min_attributes_count
        self.population_size = population_size
        self.classifier = classifier

    def _binary_representation_of_features(self, features: Set[int]) -> np.ndarray:
        binary_representation = np.zeros(Algorithm.ALL_ATTRIBUTES_COUNT, dtype=int)

        for feature in features:
            binary_representation[feature] = 1

        return binary_representation

    def _set_representation_of_features(self, individual: np.ndarray) -> Set[int]:
        return {index for index, value in enumerate(individual) if value == 1}

    def _generate_single_individual(self) -> np.ndarray:
        features_count = int(np.random.normal(self.min_attributes_count, self.initial_attributes_standard_deviation, 1))
        features = set(random_features_indices(features_count))

        return self._binary_representation_of_features(features)

    def _generate_initial_population(self) -> Population:
        return [self._generate_single_individual() for _ in range(self.population_size)]

    def _compute_fitness(self, population: Population) -> PopulationWithFitness:
        fitness_function = (self.classifier.calculate_precision_recall
                            if self.fitness_function == FitnessFunction.AveragePrecision
                            else self.classifier.calculate_area_under_roc())
        return [(fitness_function(self._set_representation_of_features(individual)), individual) for individual in
                population]

    def _selection(self, population_with_fitness: PopulationWithFitness) -> List[Tuple[int, int]]:
        return [(0, 0) for _ in range(len(population_with_fitness))]

    def _cross(self, mother: np.ndarray, father: np.ndarray) -> np.ndarray:
        loci = sorted(random_features_indices(self.loci_count - 1))
        mother_parts = np.array_split(mother, loci)
        father_parts = np.array_split(father, loci)

        consecutive_pairs = ((first, second) for first, second in
                             zip(range(0, self.loci_count, 2),
                                 range(1, self.loci_count, 2)))  # [(0,1), (2, 3), (4, 5)...]
        parts = [(mother_parts[even], father_parts[odd]) for even, odd in consecutive_pairs]

        return np.concatenate(list(itertools.chain(*parts)))

    def _crossover(self, selected_pairs: List[Tuple[int, int]], population: Population) -> Population:
        return [self._cross(population[first], population[second]) for first, second in selected_pairs]

    def _mutate(self, individual: np.ndarray) -> None:
        chromosomes_to_mutate = random_features_indices(
            int(Algorithm.ALL_ATTRIBUTES_COUNT * self.chromosomes_to_mutate_coefficient))

        for chromosome_to_mutate in chromosomes_to_mutate:
            individual[chromosome_to_mutate] = 1 if individual[chromosome_to_mutate] == 0 else 0

    def _mutation(self, population: Population) -> Population:
        individuals_to_mutate = random.sample(range(len(population)),
                                              int(len(population) * self.individuals_to_mutate_coefficient))

        for individual_index in individuals_to_mutate:
            self._mutate(population[individual_index])

        return population

    def run(self):
        population = self._generate_initial_population()
        for i in range(self.cycles_count):
            print(f"Cycle: {i}")
            population_with_fitness = self._compute_fitness(population)
            selected_pairs = self._selection(population_with_fitness)
            new_population = self._crossover(selected_pairs, population)
            self._mutation(new_population)
            population = new_population

        return self._fittest(population)

    def _fittest(self, population):
        return population[0]
