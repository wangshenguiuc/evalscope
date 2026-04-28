from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# GPQA-Extended (Idavidrein/gpqa, config `gpqa_extended`) is the
# strict superset of gpqa_main and gpqa_diamond — 546 expert-written
# graduate-level science MCQs across biology, physics, and chemistry.
# Schema (`Question`, `Correct Answer`, `Incorrect Answer 1..3`) and
# scoring (letter-match against shuffled choices) match the parent
# `gpqa_diamond` adapter exactly, so this is a thin subclass that
# only changes the BenchmarkMeta. Diamond's 198 questions are scored
# separately by the existing `gpqa_diamond` row.


@register_benchmark(
    BenchmarkMeta(
        name='gpqa_extended',
        pretty_name='GPQA-Extended',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

GPQA-Extended is the broadest config of the GPQA (Graduate-Level
Google-Proof Q&A) benchmark — 546 expert-written multiple-choice
questions in biology, physics, and chemistry. Strict superset of
gpqa_main (448) and gpqa_diamond (198).

## Task Description

- **Task Type**: Expert-Level Multiple-Choice Q&A
- **Input**: Graduate-level science question with 4 choices
- **Output**: Single correct answer letter (A, B, C, or D)
- **Domains**: Biology, Physics, Chemistry

## Evaluation Notes

- Only the train split ships with public labels.
- Choices are randomly shuffled per sample.
- Letter-match accuracy.
""",
        dataset_id='Idavidrein/gpqa',
        metric_list=['acc'],
        subset_list=['gpqa_extended'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class GPQAExtendedAdapter(GPQAAdapter):
    pass
