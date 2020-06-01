import sys

import numpy as np
import matplotlib.pyplot as plt

from data.data_reader import DataReader
from classifier.classifier import Classifier
from geneticalgorythm.algorithm import Algorithm, SelectionAlgorithm, FitnessFunction


def read_values(path):
    with open(f"{path}.data", 'r') as file:
        data = file.readlines()
        results = []
        for result in data[1:]:
            values = result.split(' ')
            results.append((int(values[0]), float(values[1])))

        return results


def plot_values(data, title, lineLabel, ylabel):
    x1 = [x[0] for x in data]
    y1 = [x[1] for x in data]

    plt.plot(x1, y1, label=lineLabel)
    plt.title(title)
    plt.xlabel('Generation number')
    plt.ylabel(ylabel)
    plt.xlim((0, 100))
    if y1[0] >= 1:
        plt.ylim((1, 1200))  # for mean attributes
    else:
        plt.ylim((0.3, 1.1)) # for rest


def plot_type_best(selection, population):
    file_name_prefix = f"Population {population}/{selection}_Fitness_precision"
    best_individuals = read_best_individuals(file_name_prefix)

    plot_values(best_individuals, 'Best individual', f'Population {population} {selection}', 'Best individual score')


def plot_type_mean(selection, population):
    file_name_prefix = f"Population {population}/{selection}_Fitness_precision"
    mean = read_mean(file_name_prefix)

    plot_values(mean, 'Population mean score', f'Population {population} {selection}', 'Population mean')


def plot_type_attrs(selection, population):
    file_name_prefix = f"Population {population}/{selection}_Fitness_precision"
    mean_attributes = read_mean_attributes(file_name_prefix)

    plot_values(mean_attributes, 'Population mean attributes count', f'Population {population} {selection}',
                'Mean attributes count')


def read_best_individuals(path):
    return read_values(f"{path}_best_individual")


def read_mean(path):
    return read_values(f"{path}_mean")


def read_mean_attributes(path):
    return read_values(f"{path}_mean_attributes")


# def read_best_individual(path):


if __name__ == '__main__':
    # Reading
    for population in [100, 700]:
        for selection in [SelectionAlgorithm.Tournament, SelectionAlgorithm.Roulette, SelectionAlgorithm.Ranking]:
            plot_type_best(selection, population)

    plt.legend()
    plt.savefig(f"combined/best")
    plt.show()
    for population in [100, 700]:
        for selection in [SelectionAlgorithm.Tournament, SelectionAlgorithm.Roulette, SelectionAlgorithm.Ranking]:
            plot_type_mean(selection, population)

    plt.legend()
    plt.savefig(f"combined/mean")
    plt.show()

    for population in [100, 700]:
        for selection in [SelectionAlgorithm.Tournament, SelectionAlgorithm.Roulette, SelectionAlgorithm.Ranking]:
            plot_type_attrs(selection, population)
    plt.legend()
    plt.savefig(f"combined/mean_attrs")
    plt.show()
