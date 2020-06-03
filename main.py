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
        plt.ylim((0.3, 1.1))  # for rest


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




def plot_best_one():
    file_name_prefix = f"Population {population}/{selection}_Fitness_precision_result"

    best_result = None
    with open(f"{file_name_prefix}.data", 'r') as file:
        data = file.readlines()
        best_result = float(data[0].split(':')[1])
        # print(sum([int(x) for x in data[1].replace('\n', '0').split(' ')]))

    return f"{population} {selection.split('_')[1]}", best_result


if __name__ == '__main__':
    # Reading
    data = []
    for population in [100, 700]:
        for selection in [SelectionAlgorithm.Tournament, SelectionAlgorithm.Roulette, SelectionAlgorithm.Ranking]:
            best_one = plot_best_one()
            data.append(best_one)
            # data[best_one[0]] = best_one[1]

    plt.ylim(0.5, 1)
    plt.bar([x[0] for x in data], [x[1] for x in data])
    plt.xticks(rotation='45')
    plt.show()
    plt.legend()
    plt.savefig(f"combined/best")
    plt.show()
