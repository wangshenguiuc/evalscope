import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

SUBSET_LIST = [
    'mcq_exams_test_ar', 'meta_ar_dialects', 'meta_ar_msa',
    'multiple_choice_facts_truefalse_balanced_task',
    'multiple_choice_grounded_statement_soqal_task',
    'multiple_choice_grounded_statement_xglue_mlqa_task',
    'multiple_choice_rating_sentiment_no_neutral_task',
    'multiple_choice_rating_sentiment_task',
    'multiple_choice_sentiment_task',
]


@register_benchmark(
    BenchmarkMeta(
        name='alghafa_arabic_native',
        pretty_name='AlGhafa Arabic LLM Benchmark — Native',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

AlGhafa Arabic LLM Benchmark — Native (`OALL/AlGhafa-Arabic-LLM-Benchmark-Native`)
is a 9-subset Arabic eval suite covering exam-style MCQ, dialect /
MSA understanding, sentiment classification, fact T/F, and grounded
QA. Each subset has a varying number of choices (2 to 5).

## Task Description

- **Task Type**: Native-Arabic Multiple-Choice (variable 2–5 choices)
- **Input**: `query`
- **Output**: Single correct answer letter (the dataset stores
  `label` as a 0-indexed string digit; the adapter maps it to A–E).

## Evaluation Notes

- 9 subsets; per-subset sizes range 75–7995.
- `sol1`/`sol2`/`...` columns exist up to whatever number of choices
  the subset uses; the adapter filters to the present columns.
- Primary metric: **Accuracy**.
""",
        dataset_id='OALL/AlGhafa-Arabic-LLM-Benchmark-Native',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AlGhafaArabicNativeAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = []
        for i in range(1, 6):
            v = record.get(f'sol{i}')
            if v is None or v == '':
                continue
            choices.append(v)
        target = string.ascii_uppercase[int(record['label'])]
        return Sample(
            input=record['query'],
            choices=choices,
            target=target,
        )
