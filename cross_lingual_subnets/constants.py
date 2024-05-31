from enum import Enum


class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class Datasets(str, ExtendedEnum):
    WIKIPEDIA = "mbelitsky/wikipedia_subset"


class Experiments(str, Enum):
    XLMR_RANDOM = "xlmr_random"
    XLMR_BASE = "xlmr_base"
    XLMR_MLM_FINETUNED = "xlmr_mlm_finetuned"
    AR_SUB_MLM_FINETUNED = "ar_sub_mlm_finetuned"
    DE_SUB_MLM_FINETUNED = "de_sub_mlm_finetuned"
    EN_SUB_MLM_FINETUNED = "en_sub_mlm_finetuned"
    ES_SUB_MLM_FINETUNED = "es_sub_mlm_finetuned"
    HI_SUB_MLM_FINETUNED = "hi_sub_mlm_finetuned"
    RU_SUB_MLM_FINETUNED = "ru_sub_mlm_finetuned"
    UR_SUB_MLM_FINETUNED = "ur_sub_mlm_finetuned"
    ZH_SUB_MLM_FINETUNED = "zh_sub_mlm_finetuned"
