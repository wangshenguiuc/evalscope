import re
import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# AGIEval-format choices ship with a leading "(A)" / "(B)" / etc. prefix
# embedded in the choice text. Strip it so the prompt template can add
# its own canonical letter prefix.
_CHOICE_PREFIX = re.compile(r'^\s*\(?[A-Da-d]\)?\s*[\.、:：]?\s*')


def _strip_choice_prefix(text: str) -> str:
    return _CHOICE_PREFIX.sub('', text, count=1).strip()


@register_benchmark(
    BenchmarkMeta(
        name='agieval_jec_qa_kd',
        pretty_name='AGIEval JEC-QA (Knowledge Driven)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

AGIEval JEC-QA (Knowledge Driven) — `hails/agieval-jec-qa-kd` — is a
1000-question Chinese legal-bar-exam MCQ split focused on items that
require legal knowledge recall (vs the reasoning-driven `jec-qa-ca`
split, also in this campaign).

## Task Description

- **Task Type**: Chinese legal MCQ
- **Input**: `query` (carries question + options + Chinese answer prompt;
  the adapter extracts the question stem and feeds canonical choices
  separately).
- **Output**: Single correct answer letter (A/B/C/D).

## Evaluation Notes

- 1000 test rows, single config `default`.
- Primary metric: **Accuracy**.
""",
        dataset_id='hails/agieval-jec-qa-kd',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AGIEvalJECQAKDAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Strip any leading "(A)" prefix from choices so the prompt
        # template's own letter-prefixing can take over cleanly.
        choices = [_strip_choice_prefix(c) for c in record['choices']]
        gold = record['gold']
        idx = gold[0] if isinstance(gold, list) else int(gold)
        target = string.ascii_uppercase[int(idx)]
        # Pull just the question stem from the embedded "问题：…选项：" block.
        query = record['query']
        m = re.search(r'问题[：:]\s*(.+?)\s*选项[：:]', query, flags=re.DOTALL)
        question = m.group(1).strip() if m else query
        return Sample(
            input=question,
            choices=choices,
            target=target,
        )
