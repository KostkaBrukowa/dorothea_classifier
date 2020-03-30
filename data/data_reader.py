from typing import List
from compound.compound import Compound


class DataReader:
    train_file = './data/data_set/dorothea_train.data'
    train_labels_file = './data/data_set/dorothea_train.labels'
    valid_file = './data/data_set/dorothea_valid.data'
    valid_labels_file = './data/data_set/dorothea_valid.labels'
    test_file = './data/data_set/dorothea_test.data'

    def _convert_active(self, label: int) -> int:
        if label == 1:
            return 1
        elif label == -1:
            return 0
        else:
            return None

    def _read_labels(self, labels_file: str) -> List[int]:
        labels = []

        with open(labels_file, 'r') as reader:
            for line in reader.readlines():
                labels.append(int(line))

        return labels

    def _read_data(self, data_file: str, labels: List[int] = None) -> List[Compound]:
        compounds = []

        with open(data_file, 'r') as reader:
            for line in enumerate(reader.readlines()):
                features = set(line[1].split())
                active = self._convert_active(
                    labels[line[0]]) if labels is not None else None
                new_compound = Compound(line[0], active, features)
                compounds.append(new_compound)

        return compounds

    def _read_full_set(self, file_labels: str, file_data: str) -> List[Compound]:
        labels = self._read_labels(file_labels)

        return self._read_data(file_data, labels)

    def read_train_data(self) -> List[Compound]:

        return self._read_full_set(self.train_labels_file, self.train_file)

    def read_valid_data(self) -> List[Compound]:

        return self._read_full_set(self.valid_labels_file, self.valid_file)

    def read_test_data(self) -> List[Compound]:

        return self._read_data(self.test_file)
