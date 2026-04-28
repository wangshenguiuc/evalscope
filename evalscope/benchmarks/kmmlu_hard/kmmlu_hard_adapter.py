from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 45 subject subsets shipped by HAERAE-HUB/KMMLU-HARD (hard subset of
# KMMLU; 100 test rows / subject; same shape as the parent KMMLU but
# config names are lowercase_underscored not Hyphenated-Capitalized).
SUBSET_LIST = [
    'maritime_engineering', 'materials_engineering',
    'railway_and_automotive_engineering', 'biology', 'public_safety',
    'criminal_law', 'information_technology', 'geomatics', 'management',
    'math', 'accounting', 'chemistry', 'nondestructive_testing',
    'computer_science', 'ecology', 'health',
    'political_science_and_sociology', 'patent',
    'electrical_engineering', 'electronics_engineering',
    'korean_history', 'gas_technology_and_engineering',
    'machine_design_and_manufacturing', 'chemical_engineering',
    'telecommunications_and_wireless_technology', 'food_processing',
    'social_welfare', 'real_estate', 'marketing',
    'mechanical_engineering', 'fashion', 'psychology', 'taxation',
    'environmental_science', 'refrigerating_machinery', 'education',
    'industrial_engineer', 'civil_engineering', 'energy_management',
    'law', 'agricultural_sciences',
    'interior_architecture_and_design',
    'aviation_engineering_and_maintenance', 'construction', 'economics',
]

_INT_TO_LETTER = ['A', 'B', 'C', 'D']


@register_benchmark(
    BenchmarkMeta(
        name='kmmlu_hard',
        pretty_name='KMMLU-HARD',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

KMMLU-HARD (HAERAE-HUB/KMMLU-HARD) is a 4500-question difficulty-
filtered subset of KMMLU (100 test rows per subject across the same
45 subjects). Same field schema as the parent KMMLU, plus a `cot`
column with a chain-of-thought trace.

## Task Description

- **Task Type**: Korean Multiple-Choice Question Answering
- **Input**: `question` plus `A`/`B`/`C`/`D` choice fields
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- 100 test + 5 dev rows per subject; 45 subsets.
- `answer` is a 1-indexed integer; mapped to A–D letter target.
- Primary metric: **Accuracy**.
""",
        dataset_id='HAERAE-HUB/KMMLU-HARD',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class KMMLUHardAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'], record['D']]
        target = _INT_TO_LETTER[int(record['answer']) - 1]
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
            metadata={
                'category': record.get('category') or self.current_subset_name,
            },
        )
