from typing import List, Tuple
from random import randint
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]


def _ranks_with_indexes(population_with_fitness: PopulationWithFitness) -> List[Tuple[int, int]]:
    sorted_population = sorted(
        population_with_fitness, key=lambda ind: ind[0])
    rank = 1
    previous_individual = sorted_population[0]
    ranks_indexes = [(rank, population_with_fitness.index(previous_individual))]
    for individual in sorted_population[1:]:
        if individual[0] > previous_individual[0]:
            rank += 1
        ranks_indexes.append((rank, population_with_fitness.index(individual)))

    return ranks_indexes


def _ranking_select_one(ranks_indexes: List[Tuple[int, int]], ranks_sum: int) -> int:
    lucky_number = randint(0, ranks_sum)
    lottery_sum = 0
    for individual in ranks_indexes:
        lottery_sum += individual[0]
        if lucky_number <= lottery_sum:
            return individual[1]


def ranking_selection(population_with_fitness: PopulationWithFitness, pair_number: int) -> List[Tuple[int, int]]:
    ranks_indexes = _ranks_with_indexes(population_with_fitness)
    ranks_sum = sum(individual[0] for individual in ranks_indexes)
    pairs = [(_ranking_select_one(ranks_indexes, ranks_sum),
              _ranking_select_one(ranks_indexes, ranks_sum)) for _ in range(pair_number)]

    return pairs
