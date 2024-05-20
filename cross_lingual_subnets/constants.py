from enum import Enum

class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

class Datasets(str, ExtendedEnum):
    WIKIPEDIA = "mbelitsky/wikipedia_subset"

class Experiments(str, Enum):
    XLMR_BASE = "xlmr_base"
    XLMR_MLM_FINETUNED = "xlmr_mlm_finetuned"

