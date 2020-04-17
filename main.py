import random

from data.data_reader import DataReader
from classifier.classifier import Classifier
from compound.compound import Compound
from geneticalgorythm.algorithm import Algorithm


def generate_compound(index: int) -> Compound:
    active = 1 if random.random() < 0.12 else 0
    characteristics = random.sample(
        range(100_000), random.randrange(500, 1000))

    return Compound(index, active, set(characteristics))


if __name__ == '__main__':
    # Reading
    data_reader = DataReader()
    train_compounds = data_reader.read_train_data()
    validation_compounds = data_reader.read_valid_data()
    test_compounds = data_reader.read_test_data()
    available_features = random.sample(
        range(100_000), random.randrange(10500, 15000))

    # Classification
    classifier = Classifier(train_compounds, validation_compounds)
    algorithm = Algorithm(classifier)

    print(algorithm.run())
    # print(classifier.predict_result(test_compounds, available_features))
