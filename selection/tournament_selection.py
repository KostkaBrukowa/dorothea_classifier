from typing import List, Tuple
from random import randrange, random
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]
WIN_PROBABILITY = 0.7


def _tournament_select_one(population_with_fitness: PopulationWithFitness, size: int) -> int:
    pair = (randrange(0, size), randrange(0, size))
    better, worse = (max(pair), min(pair))

    if random() < WIN_PROBABILITY:
        return population_with_fitness.index(better)
    else:
        return population_with_fitness.index(worse)


def tournament_selection(population_with_fitness: PopulationWithFitness, pair_number: int) -> List[Tuple[int, int]]:
    population_size = len(population_with_fitness)
    pairs = [(_tournament_select_one(population_with_fitness, population_size),
              _tournament_select_one(population_with_fitness, population_size)) for _ in
             range(pair_number)]

    return pairs
