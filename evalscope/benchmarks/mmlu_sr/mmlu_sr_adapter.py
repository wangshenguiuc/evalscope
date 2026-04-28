from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_sr',
        pretty_name='MMLU-SR',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.REASONING],
        description="""
## Overview

MMLU-SR (NiniCat/MMLU-SR — Symbol Replacement) is a contamination-
testing variant of MMLU where key terms in the question and / or
choices are replaced with placeholder symbols ('Dummy', 'Cat',
'Adam', 'Noise', etc.) along with a definition stub. Three configs:

- `answer_only` — symbol replacement applied only to choice texts
- `question_only` — symbol replacement applied only to the question
- `question_and_answer` — applied to both

## Task Description

- **Task Type**: 4-choice MCQ with symbol-replaced terms
- **Input**: `column_0` (question) plus `column_1`..`column_4` (4 choices)
- **Output**: Single correct answer letter (A/B/C/D); stored in `column_5`.

## Evaluation Notes

- 13,985 test + 228 train rows per config.
- Primary metric: **Accuracy**.
""",
        dataset_id='NiniCat/MMLU-SR',
        metric_list=['acc'],
        subset_list=['answer_only', 'question_only', 'question_and_answer'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUSRAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['column_1'], record['column_2'],
            record['column_3'], record['column_4'],
        ]
        return Sample(
            input=record['column_0'],
            choices=choices,
            target=record['column_5'],
        )
