from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 18 language subsets shipped by CohereForAI/Global-MMLU-Lite (a curated
# 200 cultural-agnostic + 200 culturally-sensitive sample subset of
# Global-MMLU per language).
SUBSET_LIST = [
    'ar', 'bn', 'cy', 'de', 'en', 'es', 'fr', 'hi', 'id',
    'it', 'ja', 'ko', 'my', 'pt', 'sq', 'sw', 'yo', 'zh',
]


@register_benchmark(
    BenchmarkMeta(
        name='global_mmlu_lite',
        pretty_name='Global-MMLU-Lite',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

Global-MMLU-Lite (CohereForAI/Global-MMLU-Lite) is a curated 400-sample
subset of Global-MMLU per language (200 culturally-agnostic + 200
culturally-sensitive items), spanning 18 languages.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: `question` plus `option_a`/`option_b`/`option_c`/`option_d`
- **Output**: Single correct answer letter (A/B/C/D)
- **Languages**: 18 subsets (see `SUBSET_LIST`)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `test` split
  (~400 rows per language; 215 dev rows for few-shot).
- Use `subset_list` to evaluate specific languages.
- Primary metric: **Accuracy**.
""",
        dataset_id='CohereForAI/Global-MMLU-Lite',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class GlobalMMLULiteAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['option_a'],
            record['option_b'],
            record['option_c'],
            record['option_d'],
        ]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={
                'sample_id': record.get('sample_id'),
                'subject': record.get('subject'),
                'subject_category': record.get('subject_category'),
                'language': self.current_subset_name,
                'cultural_sensitivity_label': record.get('cultural_sensitivity_label'),
            },
        )
