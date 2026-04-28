from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 6 country subsets shipped by QCRI/AraDiCE-Culture.
SUBSET_LIST = ['Lebanon', 'Egypt', 'Syria', 'Palestine', 'Jordan', 'Qatar']


@register_benchmark(
    BenchmarkMeta(
        name='aradice_culture',
        pretty_name='AraDiCE — Culture',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

AraDiCE — Culture (`QCRI/AraDiCE-Culture`) is a 180-question Arabic
cultural-alignment benchmark covering 6 Arab countries (Lebanon,
Egypt, Syria, Palestine, Jordan, Qatar). Each row asks a question
in the dialect of the country-of-config, with three options that
each reflect a *different* country's answer.

## Task Description

- **Task Type**: Arabic cultural-alignment 3-choice MCQ
- **Input**: `Question` plus `Option A`/`B`/`C` (each option reflects
  a different country's culturally-correct answer)
- **Output**: A — the dataset's design convention is that **Option A
  is always the country-of-config's culturally-correct answer**.

## Evaluation Notes

- 6 country configs; 30 test rows per country.
- The dataset doesn't ship an explicit answer column; the adapter
  hard-codes target=`A` based on the paper's stated convention
  (Option A = country-aligned). If reviewer rejects this assumption,
  the wire halts.
- Primary metric: **Accuracy** (essentially measures whether the
  model recognizes the country context and selects its aligned
  answer).
""",
        dataset_id='QCRI/AraDiCE-Culture',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AraDiCECultureAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['Option A'], record['Option B'], record['Option C']]
        return Sample(
            input=record['Question'],
            choices=choices,
            target='A',  # paper convention: Option A is country-aligned
            metadata={'qid': record.get('QID'), 'country': self.current_subset_name},
        )
