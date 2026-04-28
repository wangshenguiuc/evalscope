import re
import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_CHOICE_PREFIX = re.compile(r'^\s*\(?[A-Da-d]\)?\s*[\.、:：]?\s*')


def _strip_choice_prefix(text: str) -> str:
    return _CHOICE_PREFIX.sub('', text, count=1).strip()


@register_benchmark(
    BenchmarkMeta(
        name='agieval_jec_qa_ca',
        pretty_name='AGIEval JEC-QA (Case Analysis)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

AGIEval JEC-QA (Case Analysis) — `hails/agieval-jec-qa-ca` — is a
999-question Chinese legal-bar-exam **multi-select** MCQ split. Each
question has 4 options labeled A–D; the correct answer can be one or
more of them (the dataset's `gold` field is a list of correct
indices, with 1 to 4 elements per row).

## Task Description

- **Task Type**: Chinese legal multi-select MCQ
- **Input**: `query` (carries question + options + Chinese answer
  prompt; the adapter extracts the question stem and feeds canonical
  choices separately).
- **Output**: One or more correct answer letters (subset of A/B/C/D).

## Evaluation Notes

- 999 test rows, single config `default`.
- Multi-select: row distribution is 533/217/171/78 for gold-list
  sizes 1/2/3/4 respectively.
- Primary metric: **Accuracy** (exact-match against the full set of
  correct letters; no partial credit).
""",
        dataset_id='hails/agieval-jec-qa-ca',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.MULTIPLE_ANSWER,
    )
)
class AGIEvalJECQACAAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiple_correct = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [_strip_choice_prefix(c) for c in record['choices']]
        gold = record['gold']
        target = [string.ascii_uppercase[int(i)] for i in gold]
        query = record['query']
        m = re.search(r'问题[：:]\s*(.+?)\s*选项[：:]', query, flags=re.DOTALL)
        question = m.group(1).strip() if m else query
        return Sample(
            input=question,
            choices=choices,
            target=target,
        )
