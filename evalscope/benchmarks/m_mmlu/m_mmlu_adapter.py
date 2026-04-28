from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 34 language subsets shipped by alexandrainst/m_mmlu (machine-translated
# MMLU into 33 languages plus original English).
SUBSET_LIST = [
    'ar', 'bn', 'ca', 'da', 'de', 'en', 'es', 'eu', 'fr', 'gu',
    'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'kn', 'ml', 'mr',
    'nb', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta',
    'te', 'uk', 'vi', 'zh',
]


@register_benchmark(
    BenchmarkMeta(
        name='m_mmlu',
        pretty_name='m_mmlu',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

m_mmlu (alexandrainst/m_mmlu) is a multilingual MMLU translation
covering 33 languages plus English (34 subsets total). Translations
were produced via DeepL/GPT-3.5; English subset is the original MMLU.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: `instruction` plus `option_a`/`option_b`/`option_c`/`option_d`
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- 12,928 test rows per language; 1,433 val + 274 train (dev/few-shot).
- Use `subset_list` to evaluate specific languages.
- Primary metric: **Accuracy**.
""",
        dataset_id='alexandrainst/m_mmlu',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['option_a'],
            record['option_b'],
            record['option_c'],
            record['option_d'],
        ]
        return Sample(
            input=record['instruction'],
            choices=choices,
            target=record['answer'],
            metadata={
                'id': record.get('id'),
                'language': self.current_subset_name,
            },
        )
