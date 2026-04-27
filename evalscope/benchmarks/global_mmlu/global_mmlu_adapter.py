from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 42 language subsets shipped by CohereForAI/Global-MMLU.
SUBSET_LIST = [
    'am', 'ar', 'bn', 'cs', 'de', 'el', 'en', 'es', 'fa', 'fil',
    'fr', 'ha', 'he', 'hi', 'id', 'ig', 'it', 'ja', 'ko', 'ky',
    'lt', 'mg', 'ms', 'ne', 'nl', 'ny', 'pl', 'pt', 'ro', 'ru',
    'si', 'sn', 'so', 'sr', 'sv', 'sw', 'te', 'tr', 'uk', 'vi',
    'yo', 'zh',
]


@register_benchmark(
    BenchmarkMeta(
        name='global_mmlu',
        pretty_name='Global-MMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

Global-MMLU (CohereForAI/Global-MMLU) is a multilingual MMLU variant covering
42 languages with cultural-sensitivity annotations. Each language ships the
full 57-subject MMLU set plus per-question metadata flagging culture-, region-,
and country-specific knowledge requirements.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: `question` plus `option_a`/`option_b`/`option_c`/`option_d`
- **Output**: Single correct answer letter (A/B/C/D)
- **Languages**: 42 subsets (see `SUBSET_LIST`)
- **Subjects**: 57 MMLU subjects, grouped under STEM / Humanities / Social
  Sciences / Other / Business / Medical via `subject_category`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `test` split
  (~14k rows per language). A `dev` split (~285 rows per language) is also
  available for few-shot prompting if needed.
- Use `subset_list` to evaluate specific languages; full-coverage runs use
  `fast_limit` to cap volume.
- Primary metric: **Accuracy**.
""",
        dataset_id='CohereForAI/Global-MMLU',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class GlobalMMLUAdapter(MultiChoiceAdapter):

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
