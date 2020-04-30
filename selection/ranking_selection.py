from typing import List, Tuple
from random import randint
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]


def _ranks_with_indexes(populationWithFitness: PopulationWithFitness) -> List[Tuple[int, int]]:
    sorted_population = sorted(
        populationWithFitness, key=lambda individual: individual[0])
    rank = 1
    previous_individual = sorted_population[0]
    ranks_indexes = [(rank, populationWithFitness.index(previous_individual))]
    for individual in sorted_population[1:]:
        if individual[0] > previous_individual[0]:
            rank += 1
        ranks_indexes.append((rank, populationWithFitness.index(individual)))

    return ranks_indexes


def _ranking_select_one(ranks_indexes: List[Tuple[int, int]], ranks_sum: int) -> int:
    lucky_number = randint(0, ranks_sum)
    loterry_sum = 0
    for individual in ranks_indexes:
        loterry_sum += individual[0]
        if lucky_number <= loterry_sum:
            return individual[1]


def ranking_selection(populationWithFitness: PopulationWithFitness, pair_number: int) -> List[Tuple[int, int]]:
    ranks_indexes = _ranks_with_indexes(populationWithFitness)
    ranks_sum = sum(invidual[0] for invidual in ranks_indexes)
    pairs = []
    for i in range(pair_number):
        mother, father = (_ranking_select_one(ranks_indexes, ranks_sum),
                          _ranking_select_one(ranks_indexes, ranks_sum))
        pairs.append((mother, father))

    return pairs
