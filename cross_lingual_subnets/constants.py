from enum import Enum

class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

class Datasets(str, ExtendedEnum):
    EXAMPLE = "imdb"
    # TODO: add actual datasets
