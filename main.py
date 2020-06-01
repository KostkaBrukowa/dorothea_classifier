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
            generation, value = result.split(' ')
            results.append((int(generation), float(value)))

        return results


def plot_values(data, title, ylabel):
    x1 = [x[0] for x in data]
    y1 = [x[1] for x in data]

    plt.plot(x1, y1, 'o-')
    plt.title(title)
    plt.xlabel('Generation number')
    plt.ylabel(ylabel)
    plt.xlim((0, 100))
    plt.ylim((0.3, 1.1))

    plt.show()


def plot_type(selection, fitness):
    file_name_prefix = f"{selection}_{fitness}"
    best_individuals = read_best_individuals(file_name_prefix)
    mean = read_mean(file_name_prefix)
    mean_attributes = read_mean_attributes(file_name_prefix)

    plot_values(best_individuals, f'Best individual score by generation. {file_name_prefix}', 'Best individual score')
    # plot_values(mean, f'Population mean score by generation. {file_name_prefix}', 'Population mean')
    # plot_values(mean_attributes, f'Mean attributes count by generation. {file_name_prefix}', 'Mean attributes count')


def read_best_individuals(path):
    return read_values(f"{path}_best_individual")


def read_mean(path):
    return read_values(f"{path}_mean")


def read_mean_attributes(path):
    return read_values(f"{path}_mean_attributes")


# def read_best_individual(path):


if __name__ == '__main__':
    # Reading
    for fitness in [FitnessFunction.AveragePrecision, FitnessFunction.ROCCurve]:
        for selection in [SelectionAlgorithm.Tournament, SelectionAlgorithm.Roulette, SelectionAlgorithm.Ranking]:
            plot_type(selection, fitness)
