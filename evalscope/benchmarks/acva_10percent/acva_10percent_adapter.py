from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# Same 58 cultural-topic subsets as the parent OALL/ACVA, just a 10%
# subsample (~19 test rows per subset).
SUBSET_LIST = [
    'Algeria', 'Ancient_Egypt', 'Arab_Empire', 'Arabic_Architecture',
    'Arabic_Art', 'Arabic_Astronomy', 'Arabic_Calligraphy',
    'Arabic_Ceremony', 'Arabic_Clothing', 'Arabic_Culture', 'Arabic_Food',
    'Arabic_Funeral', 'Arabic_Geography', 'Arabic_History',
    'Arabic_Language_Origin', 'Arabic_Literature', 'Arabic_Math',
    'Arabic_Medicine', 'Arabic_Music', 'Arabic_Ornament',
    'Arabic_Philosophy', 'Arabic_Physics_and_Chemistry', 'Arabic_Wedding',
    'Bahrain', 'Comoros', 'Egypt_modern', 'InfluenceFromAncientEgypt',
    'InfluenceFromByzantium', 'InfluenceFromChina', 'InfluenceFromGreece',
    'InfluenceFromIslam', 'InfluenceFromPersia', 'InfluenceFromRome',
    'Iraq', 'Islam_Education', 'Islam_branches_and_schools',
    'Islamic_law_system', 'Jordan', 'Kuwait', 'Lebanon', 'Libya',
    'Mauritania', 'Mesopotamia_civilization', 'Morocco', 'Oman',
    'Palestine', 'Qatar', 'Saudi_Arabia', 'Somalia', 'Sudan', 'Syria',
    'Tunisia', 'United_Arab_Emirates', 'Yemen', 'communication',
    'computer_and_phone', 'daily_life', 'entertainment',
]

_TRUE_LABEL = 'صح'
_FALSE_LABEL = 'خطأ'
_CHOICES = [_TRUE_LABEL, _FALSE_LABEL]


@register_benchmark(
    BenchmarkMeta(
        name='acva_10percent',
        pretty_name='ACVA-10percent',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

ACVA-10percent (`arcee-globe/ACVA-10percent`) is a 10% subsample of
OALL/ACVA — Arabic True/False statements over 58 cultural / regional /
historical topics. ~19 test rows per subset.

## Task Description

- **Task Type**: Arabic True/False classification (presented as 2-choice MCQ)
- **Input**: `question` — a statement in Arabic
- **Output**: A or B — `A = صح (True)`, `B = خطأ (False)`

## Evaluation Notes

- 58 topic subsets; ~19 test rows per subset.
- Same shape as the parent `acva` row.
- Primary metric: **Accuracy**.
""",
        dataset_id='arcee-globe/ACVA-10percent',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ACVA10percentAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        ans = record['answer']
        if ans == _TRUE_LABEL:
            target = 'A'
        elif ans == _FALSE_LABEL:
            target = 'B'
        else:
            target = ''
        return Sample(
            input=record['question'],
            choices=list(_CHOICES),
            target=target,
            metadata={'id': record.get('id'), 'topic': self.current_subset_name},
        )
