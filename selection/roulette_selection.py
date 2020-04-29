from geneticalgorythm.algorithm import PopulationWithFitness
from typing import List, Tuple
from random import uniform


def roulette_selection(populationWithFitness: PopulationWithFitness, pair_number: int) -> List[Tuple[int, int]]:
    fitness_sum = sum(invidual[0] for invidual in populationWithFitness)
    pairs = []
    for i in range(pair_number):
        mother, father = (_roulette_select_one(populationWithFitness, fitness_sum),
                          _roulette_select_one(populationWithFitness, fitness_sum))
        pairs.append((mother, father))

    return pairs


def _roulette_select_one(populationWithFitness: PopulationWithFitness, fitness_sum: float) -> int:
    lucky_number = uniform(0, fitness_sum)
    loterry_sum = 0
    for individual in populationWithFitness:
        loterry_sum += individual[0]
        if lucky_number <= loterry_sum:
            return populationWithFitness.index(individual)
