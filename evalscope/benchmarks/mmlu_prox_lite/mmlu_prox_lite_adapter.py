from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 29 language subsets shipped by li-lab/MMLU-ProX-Lite.
SUBSET_LIST = [
    'af', 'ar', 'bn', 'cs', 'de', 'en', 'es', 'fr', 'hi', 'hu',
    'id', 'it', 'ja', 'ko', 'mr', 'ne', 'pt', 'ru', 'sr', 'sw',
    'te', 'th', 'uk', 'ur', 'vi', 'wo', 'yo', 'zh', 'zu',
]

_OPTION_FIELDS = [f'option_{i}' for i in range(10)]


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_prox_lite',
        pretty_name='MMLU-ProX-Lite',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

MMLU-ProX-Lite (li-lab/MMLU-ProX-Lite) is a curated 588-row test-split
subset of MMLU-ProX per language (29 languages). Same field schema as
the parent MMLU-ProX, just smaller per-language test sets.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: `question` plus `option_0`/`option_1`/.../`option_9`
  (variable length; trailing fields are null for shorter MCQs)
- **Output**: Single correct answer letter (A–J)

## Evaluation Notes

- 588 test + 70 validation rows per language; 29 language subsets.
- Primary metric: **Accuracy**.
""",
        dataset_id='li-lab/MMLU-ProX-Lite',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUProXLiteAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record[f] for f in _OPTION_FIELDS if record.get(f) is not None]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={
                'question_id': record.get('question_id'),
                'category': record.get('category'),
                'src': record.get('src'),
                'language': self.current_subset_name,
            },
        )
