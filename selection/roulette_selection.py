from typing import List, Tuple
from random import uniform
import numpy as np

PopulationWithFitness = List[Tuple[float, np.ndarray]]


def _roulette_select_one(population_with_fitness: PopulationWithFitness, fitness_sum: float) -> int:
    lucky_number = uniform(0, fitness_sum)
    lottery_sum = 0
    for index, individual in enumerate(population_with_fitness):
        lottery_sum += individual[0]
        if lucky_number <= lottery_sum:
            return index


def roulette_selection(population_with_fitness: PopulationWithFitness, pair_number: int) -> List[Tuple[int, int]]:
    fitness_sum = sum(individual[0] for individual in population_with_fitness)
    pairs = [(_roulette_select_one(population_with_fitness, fitness_sum),
              _roulette_select_one(population_with_fitness, fitness_sum)) for _ in range(pair_number)]

    return pairs
