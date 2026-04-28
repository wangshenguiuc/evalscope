from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 31 language subsets shipped by alexandrainst/m_truthfulqa
# (machine-translated TruthfulQA into 30 languages plus original
# English-style MC1/MC2 schema; the dataset doesn't ship `en`).
SUBSET_LIST = [
    'ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi',
    'hr', 'hu', 'hy', 'id', 'it', 'kn', 'ml', 'mr', 'ne', 'nl',
    'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi',
    'zh',
]


@register_benchmark(
    BenchmarkMeta(
        name='m_truthfulqa',
        pretty_name='m_truthfulqa',
        tags=[Tags.KNOWLEDGE, Tags.MULTI_LINGUAL],
        description="""
## Overview

m_truthfulqa (`alexandrainst/m_truthfulqa`) is a multilingual
machine-translated TruthfulQA covering 31 languages with the same
MC1 (single-correct) and MC2 (multiple-correct) target schemas as
the original.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Truthfulness Evaluation
- **Input**: `question` plus `mc1_targets_choices` /
  `mc1_targets_labels` (and `mc2_*` for the multi-correct variant).
- **Output**: Single correct answer letter (MC1) — selected as A
  among shuffled choices via the standard TruthfulQA scoring path.

## Evaluation Notes

- 31 language subsets; per-subset `val` split with ~770–820 rows.
- Set `multiple_correct=True` via `extra_params` to switch to MC2.
- Primary metric: **Accuracy** (multi_choice_acc).
""",
        dataset_id='alexandrainst/m_truthfulqa',
        metric_list=['multi_choice_acc'],
        subset_list=SUBSET_LIST,
        shuffle_choices=True,
        few_shot_num=0,
        train_split=None,
        eval_split='val',
        extra_params={
            'multiple_correct': {
                'type': 'bool',
                'description': 'Use multiple-answer format (MC2) if True; otherwise single-answer (MC1).',
                'value': False,
            }
        },
    )
)
class MTruthfulQaAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiple_correct = self.extra_params.get('multiple_correct', False)
        if self.multiple_correct:
            self.prompt_template = MultipleChoiceTemplate.MULTIPLE_ANSWER
        else:
            self.prompt_template = MultipleChoiceTemplate.SINGLE_ANSWER

    def record_to_sample(self, record) -> Sample:
        if not self.multiple_correct:
            choices = record['mc1_targets_choices']
            labels = record['mc1_targets_labels']
        else:
            choices = record['mc2_targets_choices']
            labels = record['mc2_targets_labels']
        target = [chr(65 + i) for i, label in enumerate(labels) if label == 1]
        return Sample(
            input=record['question'],
            choices=list(choices),
            target=target,
            metadata={
                'language': self.current_subset_name,
                'type': 'mc1' if not self.multiple_correct else 'mc2',
            },
        )
