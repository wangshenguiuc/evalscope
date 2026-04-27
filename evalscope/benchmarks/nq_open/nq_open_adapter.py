# Copyright (c) Alibaba, Inc. and its affiliates.

import re
import string
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Closed-book / open-domain — no passage. Same `ANSWER:` extraction convention
# as trivia_qa so downstream regex stays uniform across short-answer QA.
PROMPT_TEMPLATE = """
Answer the following question with a short factoid answer (a few words). The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question.

Question: {question}
""".lstrip()

# Standard NQ-Open answer normalization (Lee et al., 2019). Used for both
# prediction and reference before substring matching.
_ARTICLE_RE = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_WS_RE = re.compile(r'\s+')


def _normalize(text: str) -> str:
    if text is None:
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = _ARTICLE_RE.sub(' ', text)
    text = _WS_RE.sub(' ', text).strip()
    return text


@register_benchmark(
    BenchmarkMeta(
        name='nq_open',
        pretty_name='NQ-Open',
        dataset_id='google-research-datasets/nq_open',
        tags=[Tags.QA, Tags.KNOWLEDGE],
        description="""
## Overview

NQ-Open is the open-domain (closed-book) variant of Google's Natural Questions
benchmark. Each question is a real Google search query, and answers are short
factoid spans (entities, dates, places, names). Unlike the original NQ task,
no passage is provided — the model must answer from its parametric knowledge.

## Task Description

- **Task Type**: Open-Domain Question Answering
- **Input**: A natural-language question
- **Output**: Short factoid answer (typically 1–5 words)
- **Domain**: General world knowledge

## Key Features

- ~88K training questions and ~3.6K validation questions
- Each question has a *list* of acceptable answers (aliases / variants)
- Inclusion-based matching after Lee et al. (2019) normalization:
  prediction is correct if any normalized alias appears in the
  normalized prediction
- No supporting passage — pure knowledge recall

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **validation** split (no public test labels)
- Answers should follow the format: "ANSWER: [ANSWER]"
""",
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        metric_list=['acc'],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class NqOpenAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Stash aliases under metadata so match_score can see the original list.
        # task_state.target arrives as the joined string (Target.text does
        # ''.join), which corrupts multi-alias references like
        # ["one", "one season"] -> "oneone season".
        aliases = record['answer']
        return Sample(
            input=record['question'],
            target=aliases,
            metadata={'aliases': aliases},
        )

    def extract_answer(self, prediction: str, task_state: TaskState):
        match = re.search(r'ANSWER:\s*(.*)', prediction)
        if match:
            return match.group(1).strip()
        return prediction.strip()

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        aliases: List[str] = task_state.metadata.get('aliases') or [reference]
        pred_norm = _normalize(filtered_prediction)
        is_correct = bool(pred_norm) and any(
            _normalize(a) and _normalize(a) in pred_norm for a in aliases
        )
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        score.value = {'acc': 1.0 if is_correct else 0.0}
        score.main_score_name = 'acc'
        score.metadata = {'aliases': aliases}
        return score
