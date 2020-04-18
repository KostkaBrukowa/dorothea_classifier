import random

from data.data_reader import DataReader
from classifier.classifier import Classifier
from geneticalgorythm.algorithm import Algorithm

if __name__ == '__main__':
    # Reading
    data_reader = DataReader()
    train_compounds = data_reader.read_train_data()
    validation_compounds = data_reader.read_valid_data()

    # Classification
    classifier = Classifier(train_compounds, validation_compounds)
    algorithm = Algorithm(classifier)

    print(algorithm.run())
