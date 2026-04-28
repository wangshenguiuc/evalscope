import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 44 subject subsets shipped by MBZUAI-Paris/DarijaMMLU (Moroccan
# Arabic / Darija MMLU translation + native items).
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
        name='darijammlu',
        pretty_name='DarijaMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

DarijaMMLU (MBZUAI-Paris/DarijaMMLU) is a Moroccan-Arabic (Darija)
MCQ benchmark covering 44 subjects spanning STEM / humanities /
social science / Moroccan-specific items (driving test, civics, etc).

## Task Description

- **Task Type**: Native-Darija Multiple-Choice Question Answering
- **Input**: `question` plus `choices` (list, typically 4) and an
  optional `context` paragraph (often empty).
- **Output**: Single correct answer letter (A/B/C/D).

## Evaluation Notes

- 44 per-subject configs; per-subset test size ranges from ~70 to
  ~1400 rows.
- Primary metric: **Accuracy**.
""",
        dataset_id='MBZUAI-Paris/DarijaMMLU',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class DarijaMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = list(record['choices'])
        target = string.ascii_uppercase[int(record['answer'])]
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
                'subject_darija': record.get('subject_darija'),
                'source': record.get('source'),
            },
        )
