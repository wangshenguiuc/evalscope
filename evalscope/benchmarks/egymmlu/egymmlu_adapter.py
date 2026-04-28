import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 44 subject subsets shipped by UBC-NLP/EgyMMLU (Egyptian-Arabic MMLU
# translation; same subject schema as DarijaMMLU).
SUBSET_LIST = [
    'accounting', 'arabic_language', 'arabic_language_(general)',
    'arabic_language_(grammar)', 'biology', 'civics',
    'computer_science', 'driving_test', 'economics', 'general_knowledge',
    'geography', 'global_facts', 'high_school_european_history',
    'high_school_geography', 'high_school_government_and_politics',
    'high_school_psychology', 'high_school_statistics',
    'high_school_world_history', 'history', 'human_aging',
    'international_law', 'islamic_studies', 'jurisprudence', 'law',
    'logical_fallacies', 'management', 'management_ar', 'marketing',
    'math', 'moral_disputes', 'moral_scenarios', 'natural_science',
    'nutrition', 'philosophy', 'philosophy_ar', 'physics',
    'political_science', 'professional_law', 'professional_psychology',
    'public_relations', 'security_studies', 'social_science',
    'sociology', 'world_religions',
]


@register_benchmark(
    BenchmarkMeta(
        name='egymmlu',
        pretty_name='EgyMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

EgyMMLU (UBC-NLP/EgyMMLU) is an Egyptian-Arabic MMLU translation
covering 44 subjects (same subject schema as DarijaMMLU).

## Task Description

- **Task Type**: Egyptian-Arabic 4-choice MCQ
- **Input**: `question` plus optional `context` and `choices` list
- **Output**: Single correct answer letter (A/B/C/D); `answer` is
  0-indexed.

## Evaluation Notes

- 44 per-subject configs; per-subset test sizes vary.
- Primary metric: **Accuracy**.
""",
        dataset_id='UBC-NLP/EgyMMLU',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class EgyMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = list(record['choices'])
        ans = record.get('answer')
        if ans is None:
            target = ''
        else:
            target = string.ascii_uppercase[int(ans)]
        question = record['question']
        context = record.get('context')
        if context:
            question = f'{context}\n\n{question}'
        return Sample(
            input=question,
            choices=choices,
            target=target,
            metadata={
                'subject': record.get('subject'),
                'egy_subject': record.get('egy_subject'),
                'source': record.get('source'),
            },
        )
