from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 40 level/group/subject subsets (Egyptian-Arabic translation of
# ArabicMMLU). Format: <level>_<group>_<subject>.
SUBSET_LIST = [
    'high_humanities_history', 'high_humanities_islamic-studies',
    'high_humanities_philosophy', 'high_language_arabic-language',
    'high_social-science_civics', 'high_social-science_economics',
    'high_social-science_geography', 'high_stem_biology',
    'high_stem_computer-science', 'high_stem_physics',
    'middle_humanities_history', 'middle_humanities_islamic-studies',
    'middle_language_arabic-language',
    'middle_other_general-knowledge', 'middle_social-science_civics',
    'middle_social-science_economics',
    'middle_social-science_geography',
    'middle_social-science_social-science',
    'middle_stem_computer-science', 'middle_stem_natural-science',
    'na_humanities_islamic-studies',
    'na_language_arabic-language-general',
    'na_language_arabic-language-grammar', 'na_other_driving-test',
    'na_other_general-knowledge', 'primary_humanities_history',
    'primary_humanities_islamic-studies',
    'primary_language_arabic-language',
    'primary_other_general-knowledge',
    'primary_social-science_geography',
    'primary_social-science_social-science',
    'primary_stem_computer-science', 'primary_stem_math',
    'primary_stem_natural-science', 'prof_humanities_law',
    'univ_other_management', 'univ_social-science_accounting',
    'univ_social-science_economics',
    'univ_social-science_political-science',
    'univ_stem_computer-science',
]


@register_benchmark(
    BenchmarkMeta(
        name='aradice_arabicmmlu_egy',
        pretty_name='AraDiCE — Arabic MMLU (Egyptian)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

AraDiCE — Arabic MMLU (Egyptian dialect) (`QCRI/AraDICE-ArabicMMLU-egy`)
is the Egyptian-Arabic translation of ArabicMMLU, organized into 40
`<level>_<group>_<subject>` configs (high / middle / primary / univ /
prof / na, across humanities / social-science / stem / language /
other).

## Task Description

- **Task Type**: Egyptian-Arabic 4-or-5-choice MCQ
- **Input**: `question` (with optional `context`) plus `options` dict
  (A/B/C/D/E keys; E may be null for 4-choice items)
- **Output**: Single correct answer letter (`Answer Key`).

## Evaluation Notes

- 40 subsets; per-subset test sizes vary.
- Primary metric: **Accuracy**.
""",
        dataset_id='QCRI/AraDICE-ArabicMMLU-egy',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AraDiCEArabicMMLUEgyAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        opts = record['options']
        choices = []
        for letter in ('A', 'B', 'C', 'D', 'E'):
            v = opts.get(letter)
            if v is None or v == '':
                continue
            choices.append(v)
        question = record['question']
        context = record.get('context')
        if context:
            question = f'{context}\n\n{question}'
        return Sample(
            input=question,
            choices=choices,
            target=record['Answer Key'],
            metadata={
                'id': record.get('ID'),
                'level': record.get('Level'),
                'group': record.get('Group'),
                'subject': record.get('Subject'),
                'country': record.get('Country'),
            },
        )
