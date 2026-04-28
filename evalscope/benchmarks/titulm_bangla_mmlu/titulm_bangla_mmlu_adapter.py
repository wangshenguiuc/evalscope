from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='titulm_bangla_mmlu',
        pretty_name='TituLM Bangla MMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

TituLM Bangla MMLU (`hishab/titulm-bangla-mmlu`) is a Bangla MCQ
benchmark covering Bengali grammar/literature, Bangladeshi exam
material (university/medical/engineering admission, ICT, GK_Bangladesh),
and standard MMLU-shape subjects.

## Task Description

- **Task Type**: Bangla 4-choice MCQ
- **Input**: `question` plus `options` (list of choice strings)
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- 53 per-subject configs + an aggregate `all` config (~14.7k test rows).
- The adapter uses the aggregate `all` config to keep this row a
  single-subset wire.
- Primary metric: **Accuracy**.
""",
        dataset_id='hishab/titulm-bangla-mmlu',
        metric_list=['acc'],
        subset_list=['all'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class TituLMBanglaMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            choices=list(record['options']),
            target=record['answer'],
            metadata={'id': record.get('id')},
        )
