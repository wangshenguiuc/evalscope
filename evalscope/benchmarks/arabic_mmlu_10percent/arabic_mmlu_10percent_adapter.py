from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 57 MMLU subjects, same set as the parent OALL/Arabic_MMLU.
SUBSET_LIST = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics',
    'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions',
]


@register_benchmark(
    BenchmarkMeta(
        name='arabic_mmlu_10percent',
        pretty_name='Arabic MMLU-10percent (arcee-globe)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

Arabic MMLU-10percent (`arcee-globe/Arabic_MMLU-10percent`) — a 10%
subsample of the OALL/Arabic_MMLU translation. 57 subjects with
~10 test rows per subject.

## Task Description

- **Task Type**: Native-Arabic 4-choice MCQ
- **Input**: `question` plus literal `A`/`B`/`C`/`D` columns
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- 57 subjects, 10 test + 5 dev rows per subject.
- Primary metric: **Accuracy**.
""",
        dataset_id='arcee-globe/Arabic_MMLU-10percent',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ArabicMMLU10percentAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'], record['D']]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={'subject': record.get('subject')},
        )
