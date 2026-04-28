import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='click',
        pretty_name='CLIcK',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

CLIcK (EunsuKim/CLIcK — Cultural and Linguistic Intelligence in
Korean) is a 1,995-question multiple-choice benchmark covering Korean
cultural and linguistic knowledge across 11 categories
(KIIP economy/law/society/etc., CSAT-style language items, …).

## Task Description

- **Task Type**: Korean cultural / linguistic Multiple-Choice QA
- **Input**: `question` plus optional `paragraph` context, and a
  variable-length `choices` list (typically 4 options).
- **Output**: The selected choice. The dataset stores the answer as
  the choice **text**; the adapter maps it back to a letter.

## Evaluation Notes

- Single config `default`, single `train` split (1,995 rows). The
  authors release this as one consolidated split — no held-out test.
- Primary metric: **Accuracy**.
""",
        dataset_id='EunsuKim/CLIcK',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class CLIcKAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = list(record['choices'])
        answer_text = record['answer']
        if answer_text in choices:
            target = string.ascii_uppercase[choices.index(answer_text)]
        else:
            # Defensive fallback: the dataset is typed as text-match,
            # but if a row stores a letter or an index, accept it.
            if isinstance(answer_text, str) and len(answer_text) == 1 and answer_text.isalpha():
                target = answer_text.upper()
            elif isinstance(answer_text, int) or (isinstance(answer_text, str) and answer_text.isdigit()):
                target = string.ascii_uppercase[int(answer_text)]
            else:
                target = ''
        question = record['question']
        paragraph = record.get('paragraph')
        if paragraph:
            question = f'{paragraph}\n\n{question}'
        return Sample(
            input=question,
            choices=choices,
            target=target,
            metadata={'id': record.get('id')},
        )
