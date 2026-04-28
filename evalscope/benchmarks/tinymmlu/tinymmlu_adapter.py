from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='tinymmlu',
        pretty_name='tinyMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

tinyMMLU (tinyBenchmarks/tinyMMLU) is a 100-question subsample of MMLU
selected by Polo et al. (2024) to reproduce full-MMLU model rankings at a
fraction of the inference cost. The 100 items span 57 MMLU subjects;
items carry an `input_formatted` field with the canonical 5-shot prompt.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (zero-shot here)
- **Input**: `question` plus `choices` (list of 4 strings)
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- Single-subset eval (no per-subject breakdown — too few rows).
- `subject` retained in `metadata` for post-hoc analysis.
- Primary metric: **Accuracy**.
""",
        dataset_id='tinyBenchmarks/tinyMMLU',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class TinyMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            choices=list(record['choices']),
            target='ABCD'[int(record['answer'])],
            metadata={'subject': record.get('subject')},
        )
