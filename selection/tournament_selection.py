from typing import List, Tuple
from random import randrange, random
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]


def _tournament_select_one(populationWithFitness: PopulationWithFitness, win_probabilty: float, size: int) -> int:
    first_individual = randrange(0, size)
    second_individual = randrange(0, size)
    better, worse = (first_individual, second_individual if first_individual[0] > second_individual[0]
                     else second_individual, first_individual)

    if random() < win_probabilty:
        return populationWithFitness.index(better)
    else:
        return populationWithFitness.index(worse)


def tournament_selection(populationWithFitness: PopulationWithFitness, pair_number: int, win_probabilty: float) -> List[Tuple[int, int]]:
    population_size = len(populationWithFitness)
    pairs = []
    for i in range(pair_number):
        mother, father = (_tournament_select_one(populationWithFitness, win_probabilty, population_size),
                          _tournament_select_one(populationWithFitness, win_probabilty, population_size))
        pairs.append((mother, father))

    return pairs
