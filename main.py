import random

from classifier.classifier import Classifier
from compound.compound import Compound


def generate_compound(index: int) -> Compound:
    active = 1 if random.random() < 0.12 else 0
    characteristics = random.sample(range(100_000), random.randrange(500, 1000))

    return Compound(index, active, set(characteristics))


if __name__ == '__main__':
    # Random data
    train_compounds = [generate_compound(i) for i in range(800)]
    validation_compounds = [generate_compound(i) for i in range(350)]
    test_compounds = [generate_compound(i) for i in range(800)]
    available_features = random.sample(range(100_000), random.randrange(50, 150))

    # Classification
    classifier = Classifier(train_compounds, validation_compounds)

    print(classifier.calculate_accuracy(available_features))
    print(classifier.predict_result(test_compounds, available_features))
