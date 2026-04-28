from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 66 subject subsets shipped by ikala/tmmluplus (Traditional-Chinese
# / Taiwan-flavored MMLU+).
SUBSET_LIST = [
    'engineering_math', 'dentistry',
    'traditional_chinese_medicine_clinical_medicine', 'clinical_psychology',
    'technical', 'culinary_skills', 'mechanical', 'logic_reasoning',
    'real_estate', 'general_principles_of_law', 'finance_banking',
    'anti_money_laundering', 'ttqav2', 'marketing_management',
    'business_management', 'organic_chemistry', 'advance_chemistry',
    'physics', 'secondary_physics', 'human_behavior', 'national_protection',
    'jce_humanities', 'politic_science', 'agriculture',
    'official_document_management', 'financial_analysis', 'pharmacy',
    'educational_psychology', 'statistics_and_machine_learning',
    'management_accounting', 'introduction_to_law', 'computer_science',
    'veterinary_pathology', 'accounting', 'fire_science', 'optometry',
    'insurance_studies', 'pharmacology', 'taxation', 'trust_practice',
    'geography_of_taiwan', 'physical_education', 'auditing',
    'administrative_law', 'education_(profession_level)', 'economics',
    'veterinary_pharmacology', 'nautical_science',
    'occupational_therapy_for_psychological_disorders',
    'basic_medical_science', 'macroeconomics', 'trade',
    'chinese_language_and_literature', 'tve_design', 'junior_science_exam',
    'junior_math_exam', 'junior_chinese_exam', 'junior_social_studies',
    'tve_mathematics', 'tve_chinese_language', 'tve_natural_sciences',
    'junior_chemistry', 'music', 'education', 'three_principles_of_people',
    'taiwanese_hokkien',
]


@register_benchmark(
    BenchmarkMeta(
        name='tmmluplus',
        pretty_name='TMMLU+',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

TMMLU+ (`ikala/tmmluplus`) is a Traditional-Chinese / Taiwan-flavored
MMLU+ benchmark covering 66 subjects spanning STEM, humanities, and
Taiwan-specific topics (taiwanese_hokkien, geography_of_taiwan, etc.).

The HF audit row originally pointed at `ZoneTwelve/tmmluplus`, which
ships only the legacy dataset-script loader and is no longer
loadable under recent `datasets` versions; this row uses the
canonical upstream `ikala/tmmluplus` repo instead.

## Task Description

- **Task Type**: Traditional-Chinese 4-choice MCQ
- **Input**: `question` plus literal `A`/`B`/`C`/`D` columns
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- 66 per-subject configs; ~100 test rows per subject.
- Primary metric: **Accuracy**.
""",
        dataset_id='ikala/tmmluplus',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class TMMLUPlusAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'], record['D']]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
        )
