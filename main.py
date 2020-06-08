import sys
import random

from data.data_reader import DataReader
from classifier.classifier import Classifier
from geneticalgorythm.algorithm import Algorithm, SelectionAlgorithm, FitnessFunction


def get_selection():
    selection = sys.argv[1]
    if selection == 'selection1':
        selection = SelectionAlgorithm.Ranking
    elif selection == 'selection2':
        selection = SelectionAlgorithm.Roulette
    elif selection == 'selection3':
        selection = SelectionAlgorithm.Tournament
    else:
        raise RuntimeError("Wrong selection alorithm in program arguments")

    return selection


def get_fitness():
    fitness = sys.argv[2]
    if fitness == 'fitness1':
        fitness = FitnessFunction.ROCCurve
    elif fitness == 'fitness2':
        fitness = FitnessFunction.AveragePrecision
    else:
        raise RuntimeError("Wrong fitness alorithm in program arguments")

    return fitness


def split_compounds(train, validation):
    cut_train = []
    while len(train) > len(cut_train) * 10:
        elem = random.choice(train)
        train.remove(elem)
        cut_train.append(elem)

    cut_validation = []
    while len(validation) > len(cut_validation) * 10:
        elem = random.choice(validation)
        validation.remove(elem)
        cut_validation.append(elem)

    return train, validation, cut_train, cut_validation


if __name__ == '__main__':
    # Reading
    data_reader = DataReader()
    train_compounds = data_reader.read_train_data()
    validation_compounds = data_reader.read_valid_data()

    train, validation, cut_train, cut_validation = split_compounds(train_compounds, validation_compounds)

    # Classification
    classifier = Classifier(train, validation)
    overfit_test_classifier = Classifier(train, cut_train + cut_validation)
    for fitness in [FitnessFunction.AveragePrecision]:
        for selection in [SelectionAlgorithm.Ranking]:
            print(fitness, selection)
            algorithm = Algorithm(classifier, fitness_function=fitness, selection_algorithm=selection,
                                  overfit_test_classifier=overfit_test_classifier)

            print(algorithm.run())
