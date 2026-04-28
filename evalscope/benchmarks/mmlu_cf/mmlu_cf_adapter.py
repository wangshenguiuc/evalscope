from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_cf',
        pretty_name='MMLU-CF',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

MMLU-CF (microsoft/MMLU-CF) is a 10,000-question contamination-free
MMLU successor — items rewritten to evade pretraining-set leakage,
covering 14 subject groups (Biology / Math / Chemistry / Physics /
Law / Engineering / Other / Economics / Health / Psychology /
Business / Philosophy / Computer Science / History).

## Task Description

- **Task Type**: 4-choice MCQ
- **Input**: `Question` plus literal `A`/`B`/`C`/`D` columns
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- Single config `default`. The dataset ships an aggregate `val` split
  (10,000 rows) plus per-subject `<Subject>_val` splits for slicing.
- 5-row `dev` splits per subject for few-shot.
- Primary metric: **Accuracy**.
""",
        dataset_id='microsoft/MMLU-CF',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='val',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUCFAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'], record['D']]
        return Sample(
            input=record['Question'],
            choices=choices,
            target=record['Answer'],
        )
