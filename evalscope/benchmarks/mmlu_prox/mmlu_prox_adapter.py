from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 29 language subsets shipped by li-lab/MMLU-ProX.
SUBSET_LIST = [
    'af', 'ar', 'bn', 'cs', 'de', 'en', 'es', 'fr', 'hi', 'hu',
    'id', 'it', 'ja', 'ko', 'mr', 'ne', 'pt', 'ru', 'sr', 'sw',
    'te', 'th', 'uk', 'ur', 'vi', 'wo', 'yo', 'zh', 'zu',
]

_OPTION_FIELDS = [f'option_{i}' for i in range(10)]


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_prox',
        pretty_name='MMLU-ProX',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

MMLU-ProX (li-lab/MMLU-ProX) is a multilingual extension of MMLU-Pro across
29 languages. Each language ships ~11.8k test rows expanded toward MMLU-Pro's
10-option format; questions inherited from the original 4-option MMLU keep
their first four options and pad `option_4`..`option_9` with null.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: `question` plus `option_0`/`option_1`/.../`option_9` (variable
  length; trailing fields are null for shorter MCQs)
- **Output**: Single correct answer letter (A–J)
- **Languages**: 29 subsets (see `SUBSET_LIST`)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `test` split
  (~11.8k rows per language). A `validation` split (70 rows per language)
  is available for few-shot prompting.
- Use `subset_list` to evaluate specific languages; full-coverage runs use
  `fast_limit` to cap volume.
- Primary metric: **Accuracy**.
""",
        dataset_id='li-lab/MMLU-ProX',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUProXAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Filter trailing-null option fields. Order is preserved so the gold
        # letter (A=option_0, B=option_1, …) still indexes the correct
        # choice — trailing nulls only appear past the gold answer.
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
