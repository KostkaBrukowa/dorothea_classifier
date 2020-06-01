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


def plot_values(data, title, ylabel):
    x1 = [x[0] for x in data]
    y1 = [x[1] for x in data]

    plt.plot(x1, y1, 'o-')
    plt.title(title)
    plt.xlabel('Generation number')
    plt.ylabel(ylabel)
    plt.xlim((0, 100))
    plt.ylim((0.3, 1.1))
    # plt.ylim((1, 1200)) for mean attributes



def plot_type(selection, population):
    file_name_prefix = f"Population {population}/{selection}_Fitness_precision"
    best_individuals = read_best_individuals(file_name_prefix)
    mean = read_mean(file_name_prefix)
    mean_attributes = read_mean_attributes(file_name_prefix)

    plot_values(best_individuals, f'Best individual score by generation.\n {file_name_prefix}', 'Best individual score')
    plt.savefig(f"best_images/Population{population}/{selection}")
    plt.show()

    plot_values(mean, f'Population mean score by generation.\n {file_name_prefix}', 'Population mean')
    plt.savefig(f"mean_images/Population{population}/{selection}")
    plt.show()

    plot_values(mean_attributes, f'Mean attributes count by generation.\n {file_name_prefix}', 'Mean attributes count')
    plt.savefig(f"attrs_images/Population{population}/{selection}")
    plt.show()


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
            plot_type(selection, population)
