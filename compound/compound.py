from typing import Set


class Compound:
    index: int
    active: int  # 1 if active 0 if inactive None if unknown
    features: Set[int]

    def __init__(self, index, active, features):
        self.features = features
        self.active = active
        self.index = index

    def has_feature(self, characteristic_index: int) -> bool:
        return characteristic_index in self.features
