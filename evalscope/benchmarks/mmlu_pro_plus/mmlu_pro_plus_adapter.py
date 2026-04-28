from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_pro_plus',
        pretty_name='MMLU-Pro+',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

MMLU-Pro+ (`saeidasgari/mmlu-pro-plus`) is an MMLU-Pro variant —
12,032 test rows; same 10-option MCQ shape as MMLU-Pro, with
contamination-controlled question rephrasings.

## Task Description

- **Task Type**: 4–10 choice MCQ
- **Input**: `question` plus `options` (list of choice strings)
- **Output**: Single correct answer letter (A–J)

## Evaluation Notes

- Single config `default`. 12,032 test + 70 validation rows.
- Primary metric: **Accuracy**.
""",
        dataset_id='saeidasgari/mmlu-pro-plus',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUProPlusAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [v for v in record['options'] if v is not None and v != '']
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={
                'question_id': record.get('question_id'),
                'category': record.get('category'),
                'src': record.get('src'),
            },
        )
