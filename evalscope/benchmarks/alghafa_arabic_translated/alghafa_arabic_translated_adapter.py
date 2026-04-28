import string
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 11 task subsets shipped by OALL/AlGhafa-Arabic-LLM-Benchmark-Translated
# (Arabic translations of common English benchmarks: ARC,
# HellaSwag-Okapi, MMLU-Okapi, OpenBookQA, PIQA, RACE, SciQ, COPA,
# BoolQ, Toxigen). Schema **drifts** across subsets:
#   - common shape: query + sol1..solN + int label
#   - boolq_ar: question + passage + bool answer  (no choices column)
#   - copa_ext_ar: premise + choice1 + choice2 + question + label
SUBSET_LIST = [
    'arc_challenge_okapi_ar', 'arc_easy_ar', 'boolq_ar', 'copa_ext_ar',
    'hellaswag_okapi_ar', 'mmlu_okapi_ar', 'openbook_qa_ext_ar',
    'piqa_ar', 'race_ar', 'sciq_ar', 'toxigen_ar',
]

# Arabic True/False labels (used for boolq_ar synthetic choices).
_TRUE_LABEL = 'صح'
_FALSE_LABEL = 'خطأ'


@register_benchmark(
    BenchmarkMeta(
        name='alghafa_arabic_translated',
        pretty_name='AlGhafa Arabic LLM Benchmark — Translated',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

AlGhafa Arabic LLM Benchmark — Translated
(`OALL/AlGhafa-Arabic-LLM-Benchmark-Translated`) is a 12-subset
Arabic translation of common English benchmarks (ARC,
HellaSwag-Okapi, MMLU-Okapi, OpenBookQA, PIQA, RACE, SciQ, COPA,
BoolQ, Toxigen).

## Task Description

- **Task Type**: Arabic-translated MCQ (variable choice count, mostly 4)
- **Input**: `query` (or `question` + `passage` for BoolQ;
  `premise` + `question` for COPA) plus 2–4 choice columns
- **Output**: Single correct answer letter (the dataset stores
  varied target schemas; the adapter normalizes).

## Evaluation Notes

- 11 subsets; per-subset sizes range 90–4929.
- Three distinct internal schemas (`solN`+label, BoolQ, COPA).
- Primary metric: **Accuracy**.
""",
        dataset_id='OALL/AlGhafa-Arabic-LLM-Benchmark-Translated',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AlGhafaArabicTranslatedAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Detect shape by present columns.
        if 'sol1' in record:
            choices: List[str] = []
            for i in range(1, 6):
                v = record.get(f'sol{i}')
                if v is None or v == '':
                    continue
                choices.append(v)
            label = record['label']
            target = string.ascii_uppercase[int(label)]
            question = record['query']
        elif 'choice1' in record and 'choice2' in record:
            # COPA shape: premise + question (cause/effect) + 2 choices
            choices = [record['choice1'], record['choice2']]
            target = string.ascii_uppercase[int(record['label'])]
            question = f"{record['premise']}\n\n{record.get('question', '')}".strip()
        elif 'passage' in record and 'answer' in record:
            # BoolQ shape: passage + question + bool/string answer
            choices = [_TRUE_LABEL, _FALSE_LABEL]
            ans = record['answer']
            if isinstance(ans, bool):
                target = 'A' if ans else 'B'
            elif isinstance(ans, str):
                target = 'A' if ans.lower() in ('true', 'yes', _TRUE_LABEL) else 'B'
            elif isinstance(ans, int):
                target = 'A' if ans == 1 else 'B'
            else:
                target = ''
            question = f"{record['passage']}\n\n{record['question']}"
        else:
            # Fallback — should not happen
            choices = []
            target = ''
            question = str(record)
        return Sample(
            input=question,
            choices=choices,
            target=target,
            metadata={'subset': self.current_subset_name},
        )
