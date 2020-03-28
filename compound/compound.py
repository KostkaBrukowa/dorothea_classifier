from dataclasses import dataclass
from typing import Set


@dataclass
class Compound:
    index: int
    active: int  # 1 if active 0 if inactive None if unknown
    features: Set[int]

    def has_feature(self, characteristic_index: int) -> bool:
        return characteristic_index in self.features
