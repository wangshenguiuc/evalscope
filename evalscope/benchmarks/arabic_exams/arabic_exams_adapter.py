from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='arabic_exams',
        pretty_name='Arabic Exams (OALL)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

Arabic Exams (`OALL/Arabic_EXAMS`) is a 537-row test of Arabic
high-school exam items spanning Islamic Studies, Sciences, History,
Geography, and other Arab-region school subjects.

## Task Description

- **Task Type**: Native-Arabic 4-choice MCQ
- **Input**: `question` plus literal `A`/`B`/`C`/`D` columns
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- Single config `default`. 537 test + 25 validation rows.
- Primary metric: **Accuracy**.
""",
        dataset_id='OALL/Arabic_EXAMS',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ArabicExamsAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'], record['D']]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={'id': record.get('id'), 'subject': record.get('subject')},
        )
