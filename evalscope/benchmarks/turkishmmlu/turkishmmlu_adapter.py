import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='turkishmmlu',
        pretty_name='TurkishMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

TurkishMMLU (AYueksel/TurkishMMLU) is a 5-choice native-Turkish MMLU
covering 9 subjects (Biology, Geography, Chemistry, History,
Mathematics, Philosophy, Physics, Religion/Ethics, Turkish Language
and Literature) plus an aggregate `All` config (900 test rows).

## Task Description

- **Task Type**: Native-Turkish 5-choice MCQ
- **Input**: `question` plus `choices` (list of 5)
- **Output**: Single correct answer letter (A/B/C/D/E)

## Evaluation Notes

- 9 per-subject configs (100 test/subject) + 1 aggregate `All`
  config (900 rows). The adapter uses `All` as the sole subset.
- Primary metric: **Accuracy**.
""",
        dataset_id='AYueksel/TurkishMMLU',
        metric_list=['acc'],
        subset_list=['All'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class TurkishMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        ans = record['answer']
        # `All` config stores answer as a 0-indexed int; per-subject configs
        # store the canonical letter. Coerce to letter for the parent's
        # letter-match scoring.
        if isinstance(ans, int):
            target = string.ascii_uppercase[ans]
        elif isinstance(ans, str) and ans.isdigit():
            target = string.ascii_uppercase[int(ans)]
        else:
            target = str(ans)
        return Sample(
            input=record['question'],
            choices=list(record['choices']),
            target=target,
        )
