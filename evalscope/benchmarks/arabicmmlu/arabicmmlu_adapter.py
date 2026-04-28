import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='arabicmmlu',
        pretty_name='ArabicMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

ArabicMMLU (MBZUAI/ArabicMMLU) is a native Arabic MMLU built from
exam-bank items across 8 Arabic-speaking countries — 14,455 test
items spanning 40 subjects across STEM / Humanities / Social Science /
Arabic Language / Other.

## Task Description

- **Task Type**: Native-Arabic Multiple-Choice Question Answering
- **Input**: `Question` (+ optional `Context`) plus 4 or 5 options
  under `Option 1`..`Option 5` (some questions are 4-way; `Option 5`
  is null in those rows).
- **Output**: Single correct answer letter (A/B/C/D[/E])

## Evaluation Notes

- 14,455 test rows under config `All`; 120 dev rows for few-shot.
- Per-subject configs also exist but are subsumed by `All`.
- Primary metric: **Accuracy**.
""",
        dataset_id='MBZUAI/ArabicMMLU',
        metric_list=['acc'],
        subset_list=['All'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ArabicMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = []
        for i in range(1, 6):
            opt = record.get(f'Option {i}')
            if opt is not None and opt != '':
                choices.append(opt)
        question = record['Question']
        context = record.get('Context')
        if context:
            question = f'{context}\n\n{question}'
        target = record['Answer Key']
        if target.isdigit():
            target = string.ascii_uppercase[int(target) - 1]
        return Sample(
            input=question,
            choices=choices,
            target=target,
            metadata={
                'id': record.get('ID'),
                'group': record.get('Group'),
                'subject': record.get('Subject'),
                'level': record.get('Level'),
                'country': record.get('Country'),
            },
        )
