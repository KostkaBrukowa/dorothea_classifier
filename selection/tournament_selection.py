from typing import List, Tuple
from random import randrange, random
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]
WIN_PROBABILITY = 0.7


def _tournament_select_one(population_with_fitness: PopulationWithFitness, size: int) -> int:
    first_index = randrange(0, size)
    second_index = randrange(0, size)

    better = first_index if population_with_fitness[first_index][0] > population_with_fitness[second_index][0]\
        else second_index
    worse = first_index if better == second_index else second_index

    if random() < WIN_PROBABILITY:
        return better
    else:
        return worse


def tournament_selection(population_with_fitness: PopulationWithFitness, pair_number: int) -> List[Tuple[int, int]]:
    population_size = len(population_with_fitness)
    pairs = [(_tournament_select_one(population_with_fitness, population_size),
              _tournament_select_one(population_with_fitness, population_size)) for _ in
             range(pair_number)]

    return pairs
