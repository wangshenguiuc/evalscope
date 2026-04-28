import ast
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 18 language subsets shipped by masakhane/afrimmlu (translated MMLU
# into 16 African languages plus English and French).
SUBSET_LIST = [
    'amh', 'eng', 'ewe', 'fra', 'hau', 'ibo', 'kin', 'lin', 'lug',
    'orm', 'sna', 'sot', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul',
]


@register_benchmark(
    BenchmarkMeta(
        name='afrimmlu',
        pretty_name='AfriMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

AfriMMLU (masakhane/afrimmlu) is a translation of MMLU into 16 African
languages, plus English and French (18 subsets total). Built by the
Masakhane research collective.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: `question` plus `choices` (Python-list literal of 4 options)
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- 500 test + 83 validation + 25 dev rows per language.
- `choices` ships as a stringified Python list — the adapter parses it
  with `ast.literal_eval`.
- Primary metric: **Accuracy**.
""",
        dataset_id='masakhane/afrimmlu',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AfriMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        raw = record['choices']
        if isinstance(raw, list):
            choices = [str(c) for c in raw]
        else:
            try:
                parsed = ast.literal_eval(str(raw))
                choices = [str(c) for c in parsed] if isinstance(parsed, (list, tuple)) else [str(raw)]
            except (ValueError, SyntaxError):
                choices = [str(raw)]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={
                'subject': record.get('subject'),
                'language': self.current_subset_name,
            },
        )
