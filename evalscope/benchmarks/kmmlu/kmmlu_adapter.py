from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 45 subject subsets shipped by HAERAE-HUB/KMMLU.
SUBSET_LIST = [
    'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance',
    'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering',
    'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics',
    'Education', 'Electrical-Engineering', 'Electronics-Engineering',
    'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
    'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer',
    'Information-Technology', 'Interior-Architecture-and-Design', 'Korean-History',
    'Law', 'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering',
    'Marketing', 'Materials-Engineering', 'Math', 'Mechanical-Engineering',
    'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology',
    'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering',
    'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation',
    'Telecommunications-and-Wireless-Technology',
]

_INT_TO_LETTER = ['A', 'B', 'C', 'D']


@register_benchmark(
    BenchmarkMeta(
        name='kmmlu',
        pretty_name='KMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

KMMLU (HAERAE-HUB/KMMLU) is a Korean MMLU-style benchmark covering 45 subjects
with 35,030 test questions sourced from original Korean exams (not translations).
Subjects span HUMSS, STEM, Applied Science, and Other supercategories.

## Task Description

- **Task Type**: Korean Multiple-Choice Question Answering
- **Input**: `question` plus `A`/`B`/`C`/`D` choice fields
- **Output**: Single correct answer letter (A/B/C/D)
- **Subjects**: 45 (`Accounting`, `Math`, `Korean-History`, ...)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `test` split.
  A `dev` split (5 per subject) is available for few-shot prompting.
- The dataset stores the answer as an integer 1-4; this adapter maps it
  to the canonical `A`-`D` letter target expected by `MultiChoiceAdapter`.
- Primary metric: **Accuracy**.
""",
        dataset_id='HAERAE-HUB/KMMLU',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class KMMLUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'], record['D']]
        target = _INT_TO_LETTER[int(record['answer']) - 1]
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
            metadata={
                'subject': record.get('Category') or self.current_subset_name,
                'human_accuracy': record.get('Human Accuracy'),
            },
        )
