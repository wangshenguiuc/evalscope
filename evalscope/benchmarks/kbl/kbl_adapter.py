from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 67 subsets shipped by lbox/kbl: 11 kbl_knowledge_*/kbl_reasoning_*
# subsets (2–3 choices each) + 56 bar_exam_<category>_<year> subsets
# (5 choices each).  We cap to a representative slice at smoke time
# via the scheduler's subset_list cap; the registry includes all 67.
SUBSET_LIST = [
    'kbl_knowledge_common_legal_mistake_qa',
    'kbl_knowledge_common_legal_mistake_qa_reasoning',
    'kbl_knowledge_legal_concept_qa',
    'kbl_knowledge_offense_component_qa',
    'kbl_knowledge_query_and_statute_matching_qa',
    'kbl_knowledge_statute_hallucination_qa',
    'kbl_knowledge_statute_number_and_content_matching_qa',
    'kbl_reasoning_case_relevance_qa_p',
    'kbl_reasoning_case_relevance_qa_q',
    'kbl_reasoning_causal_reasoning_qa',
    'kbl_reasoning_statement_consistency_qa',
] + [
    f'bar_exam_{cat}_{yr}'
    for cat in ('civil', 'criminal', 'public')
    for yr in range(2012, 2026)
] + [
    f'bar_exam_responsibility_{yr}' for yr in range(2010, 2024)
]

_LETTERS = ['A', 'B', 'C', 'D', 'E']


@register_benchmark(
    BenchmarkMeta(
        name='kbl',
        pretty_name='KBL (Korean Benchmark for Legal Language Understanding)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

KBL (lbox/kbl) is a Korean legal-domain MCQ benchmark covering 67
subsets — 11 knowledge / reasoning probes (2–3 choices each) plus
56 bar-exam shards (5 choices each) for civil/criminal/public/
professional-responsibility law across 2010–2025.

## Task Description

- **Task Type**: Korean legal MCQ
- **Input**: `question` plus 2–5 letter-keyed choice columns
  (`A`/`B`/`C`/`D`/`E`)
- **Output**: Single correct answer letter; the dataset stores the
  target as either `label` (knowledge/reasoning subsets) or `gt`
  (bar-exam subsets). The adapter handles both.

## Evaluation Notes

- 67 subsets total; per-subset sizes range from ~30 to ~300 rows.
- Choice count varies: 2 (case-relevance pairs), 3 (knowledge MCQ),
  5 (bar-exam).
- Primary metric: **Accuracy**.
""",
        dataset_id='lbox/kbl',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class KBLAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices: List[str] = []
        for letter in _LETTERS:
            v = record.get(letter)
            if v is None or v == '':
                continue
            choices.append(v)
        target = record.get('label') or record.get('gt') or ''
        if isinstance(target, str):
            target = target.strip().upper()
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
        )
