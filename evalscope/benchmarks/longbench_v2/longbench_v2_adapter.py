# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Tuple

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import Choices
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import answer_options, format_letter_choices

logger = get_logger()

# LongBench-v2 单条 context 可达百万字以上；多数 API/模型有 max_context（如 256k tokens）。
# 超过时 SGLang/OpenAI 会 400。默认对 document 做头尾保留截断，避免整评测失败。
# 覆盖优先级：环境变量 LONGBENCH_V2_MAX_DOCUMENT_CHARS > dataset_args.longbench_v2.max_document_chars > 默认。
_DEFAULT_MAX_DOCUMENT_CHARS = 380_000
_ENV_MAX_DOCUMENT_CHARS = 'LONGBENCH_V2_MAX_DOCUMENT_CHARS'


def _truncate_context_head_tail(text: str, max_chars: int) -> Tuple[str, bool]:
    """Keep head + tail so long-doc QA still sees beginning and end; middle is dropped."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    sep = (
        '\n\n[... middle of document omitted to fit model context limit; '
        'set LONGBENCH_V2_MAX_DOCUMENT_CHARS or dataset_args max_document_chars to adjust ...]\n\n'
    )
    budget = max_chars - len(sep)
    if budget < 2000:
        return text[:max_chars], True
    head = budget // 2
    tail = budget - head
    return text[:head] + sep + text[-tail:], True


LONGBENCH_V2_TEMPLATE = r"""Please read the following text and answer the questions below.

<text>
{document}
</text>

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}""".strip()


@register_benchmark(
    BenchmarkMeta(
        name='longbench_v2',
        pretty_name='LongBench-v2',
        tags=[Tags.READING_COMPREHENSION, Tags.MULTIPLE_CHOICE, Tags.LONG_CONTEXT],
        description="""
## Overview

LongBench v2 is a challenging benchmark for evaluating long-context understanding of large language models. It covers a wide variety of real-world tasks that require reading and comprehending long documents (ranging from a few thousand to over 2 million tokens), spanning multiple domains such as single-document QA, multi-document QA, long in-context learning, long-structured data understanding, and code repository understanding.

## Task Description

- **Task Type**: Long-Context Multiple-Choice Question Answering
- **Input**: Long document context + multiple-choice question with four answer choices (A, B, C, D)
- **Output**: Single correct answer letter
- **Domains**: Single-Doc QA, Multi-Doc QA, Long In-Context Learning, Long Structured Data Understanding, Code Repo Understanding
- **Difficulty**: Easy / Hard
- **Length**: Short / Medium / Long

## Key Features

- 503 high-quality questions requiring genuine long-document understanding
- Context lengths ranging from a few thousand tokens to over 2 million tokens
- Questions are bilingual (English and Chinese)
- Designed to require careful reading; correct answers cannot be guessed without reading the document
- Covers diverse real-world application scenarios

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (train split used as test set)
- Primary metric: **Accuracy** (exact match on letter choice)
- All four answer choices are required; no random shuffling needed
- Samples are split into **3 subsets by context length**: `short`, `medium`, `long`
- Use `subset_list` to evaluate specific length subsets (e.g., `['short', 'medium']`)
""",
        dataset_id='ZhipuAI/LongBench-v2',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        subset_list=['short', 'medium', 'long'],
        prompt_template=LONGBENCH_V2_TEMPLATE,
    )
)
class LongBenchV2Adapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True  # Split samples by 'length' field into short/medium/long subsets

    def _max_document_chars(self) -> int:
        env = os.environ.get(_ENV_MAX_DOCUMENT_CHARS, '').strip()
        if env:
            try:
                return max(0, int(env, 10))
            except ValueError:
                logger.warning('Invalid %s=%r, using default', _ENV_MAX_DOCUMENT_CHARS, env)
        if self._task_config and getattr(self._task_config, 'dataset_args', None):
            da = self._task_config.dataset_args.get(self.name, {}) or {}
            v = da.get('max_document_chars')
            if v is not None:
                try:
                    return max(0, int(v))
                except (TypeError, ValueError):
                    logger.warning('Invalid max_document_chars=%r, using default', v)
        return _DEFAULT_MAX_DOCUMENT_CHARS

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['choice_A'],
            record['choice_B'],
            record['choice_C'],
            record['choice_D'],
        ]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],  # already a letter: 'A', 'B', 'C', or 'D'
            subset_key=record.get('length', 'short'),  # Used by reformat_subset to split into short/medium/long
            metadata={
                'domain': record.get('domain', ''),
                'sub_domain': record.get('sub_domain', ''),
                'difficulty': record.get('difficulty', ''),
                'length': record.get('length', ''),
                'context': record.get('context', ''),
                '_id': record.get('_id', ''),
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        context = sample.metadata.pop('context', '') if sample.metadata else ''
        max_doc = self._max_document_chars()
        if max_doc > 0:
            context, truncated = _truncate_context_head_tail(context, max_doc)
            if truncated:
                logger.debug(
                    'LongBench-v2: context truncated to max_document_chars=%s (sample_id=%s)',
                    max_doc,
                    sample.metadata.get('_id', '') if sample.metadata else '',
                )
        choices = Choices(sample.choices)
        choices_text = answer_options(choices)
        letters = format_letter_choices(choices)

        return self.prompt_template.format(
            document=context,
            question=sample.input,
            choices=choices_text,
            letters=letters,
        )
