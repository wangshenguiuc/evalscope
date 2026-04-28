import ast
import re
import string
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 11 MCQ subsets. lyrics_denoising / proverbs_denoising are excluded —
# their `options` field is `'nan'`, marking them as free-form generation
# tasks where the model produces the restored text directly. Out of
# scope for the MCQ-only adapter shape; would need a separate
# generative/judged wire.
SUBSET_LIST = [
    'correct_definition_matching', 'csat_geo', 'csat_law', 'csat_socio',
    'date_understanding', 'general_knowledge', 'history', 'loan_words',
    'rare_words', 'standard_nomenclature', 'reading_comprehension',
]

_ANSWER_LETTER_STRICT = re.compile(r'^\s*\(?\s*([A-Ea-e])\s*\)?\s*$')


def _parse_options(opt_field: str) -> List[str]:
    """HAE-RAE-BENCH ships `options` as one of two text shapes:
    - a Python-list literal: "['opt1', 'opt2', ...]"
    - a `|`-delimited string (csat_geo): " (A), (B)  |  (C), (D)  | ..."
    """
    if not isinstance(opt_field, str):
        return list(opt_field)
    s = opt_field.strip()
    if s.startswith('['):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in s.split('|') if part.strip()]


def _question_stem(query: str) -> str:
    """Strip the embedded `### 선택지:` / `### 정답:` block; the prompt
    template re-renders choices and the answer prompt canonically."""
    if not query:
        return query
    m = re.search(r'(?s)^(.*?)\n###\s*선택지\s*[:：]', query)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?s)^(.*?)\n###\s*정답\s*[:：]', query)
    if m:
        return m.group(1).strip()
    return query.strip()


@register_benchmark(
    BenchmarkMeta(
        name='hae_rae_bench',
        pretty_name='HAE-RAE-Bench',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

HAE-RAE-Bench (HAERAE-HUB/HAE_RAE_BENCH) is a Korean linguistic and
cultural benchmark with 13 subsets covering definitions, lyrics
denoising, CSAT-style QA (geography/law/sociology), reading
comprehension, history, date understanding, and more.

## Task Description

- **Task Type**: Korean cultural / linguistic Multiple-Choice QA
- **Input**: `query` (carries question stem + options block);
  the adapter strips the options block and feeds canonical choices.
- **Output**: Single correct answer letter (A/B/C/D/E).

## Evaluation Notes

- 13 subsets; per-subset test sizes vary from ~150 to ~1000.
- `options` ships as either a Python-list literal or a `|`-delimited
  string (csat_geo); the adapter detects and parses both.
- Primary metric: **Accuracy**.
""",
        dataset_id='HAERAE-HUB/HAE_RAE_BENCH',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class HAERAEBenchAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = _parse_options(record['options'])
        ans = record['answer']
        target = ''
        if isinstance(ans, int):
            target = string.ascii_uppercase[ans]
        elif isinstance(ans, str):
            # Try strict letter form: "(A)" / "A".
            m = _ANSWER_LETTER_STRICT.match(ans)
            if m:
                target = m.group(1).upper()
            elif ans in choices:
                # Denoising subsets store the choice text rather than a letter.
                target = string.ascii_uppercase[choices.index(ans)]
        question = _question_stem(record['query'])
        return Sample(
            input=question,
            choices=choices,
            target=target,
        )
