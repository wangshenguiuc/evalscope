from __future__ import annotations

from typing import Any, Dict
import re

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

# 与 hmmt_nov_25 一致：最后一行 ANSWER: ...
PROMPT_TEMPLATE = """
Solve the following math competition problem step by step.

The last line of your response must be exactly in the format:
ANSWER: <final_answer>

Do not use \\boxed{{}}.
Do not add extra text after the final answer line.

Problem:
{question}

Reasoning:
""".lstrip()


@register_benchmark(
    BenchmarkMeta(
        name='imo_answer_bench',
        pretty_name='IMO-AnswerBench',
        dataset_id='OpenEvals/IMO-AnswerBench',
        tags=[Tags.MATH, Tags.REASONING],
        description='IMO-AnswerBench (OpenEvals/HF); short-answer olympiad problems.',
        subset_list=['default'],
        few_shot_num=0,
        train_split='train',
        eval_split='train',
        metric_list=['acc'],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class IMOAnswerBenchAdapter(DefaultDataAdapter):
    """
    HF 字段（见数据集卡）:
      Problem ID, Problem, Short Answer, Category, Subcategory, Source
    """

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record['Problem']).strip()
        target = self._normalize_answer(str(record['Short Answer']))
        cat = record.get('Category') or 'default'
        subset_key = str(cat).strip().lower() or 'default'

        return Sample(
            input=problem,
            target=target,
            subset_key=subset_key,
            metadata={
                'problem_id': record.get('Problem ID'),
                'category': record.get('Category'),
                'subcategory': record.get('Subcategory'),
                'source': record.get('Source'),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState):
        if prediction is None:
            return ''

        text = prediction.strip()
        m = re.search(r'ANSWER\s*:\s*(.+)', text, flags=re.IGNORECASE)
        if m:
            return self._normalize_answer(m.group(1).strip())

        last_line = text.splitlines()[-1].strip() if text.splitlines() else text
        return self._normalize_answer(last_line)

    @staticmethod
    def _normalize_answer(ans: str) -> str:
        if ans is None:
            return ''

        s = str(ans).strip()
        if len(s) >= 2 and s[0] == '$' and s[-1] == '$':
            s = s[1:-1].strip()

        boxed_match = re.fullmatch(r'\\boxed\{(.+)\}', s)
        if boxed_match:
            s = boxed_match.group(1).strip()

        s = re.sub(r'\s+', ' ', s).strip()
        s = s.rstrip('.').strip()
        return s
