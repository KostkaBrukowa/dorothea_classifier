import sys

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


if __name__ == '__main__':
    # Reading
    data_reader = DataReader()
    train_compounds = data_reader.read_train_data()
    validation_compounds = data_reader.read_valid_data()

    # Classification
    classifier = Classifier(train_compounds, validation_compounds)
    for fitness in [FitnessFunction.ROCCurve]:
        for selection in [SelectionAlgorithm.Ranking]:
            print(fitness, selection)
            algorithm = Algorithm(classifier, fitness_function=fitness, selection_algorithm=selection)

            print(algorithm.run())
