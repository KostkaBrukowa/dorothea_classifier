from typing import List, Tuple
from random import randint
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]


def _ranks_with_indexes(population_with_fitness: PopulationWithFitness) -> List[Tuple[int, int]]:
    sorted_population_with_indices = sorted(
        enumerate(population_with_fitness), key=lambda ind: ind[1][0])
    rank = 1
    previous_individual = sorted_population_with_indices[0]
    ranks_indexes = [(rank, previous_individual[0])]
    for individual in sorted_population_with_indices[1:]:
        if individual[1][0] > previous_individual[1][0]:
            rank += 1
        ranks_indexes.append((rank, individual[0]))

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
