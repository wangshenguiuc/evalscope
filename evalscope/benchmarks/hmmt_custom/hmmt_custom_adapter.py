from __future__ import annotations

from typing import Any, Dict
import re

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

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
        name='hmmt_nov_25',
        pretty_name='HMMT_Nov_25',
        # HuggingFace 数据集（非 ModelScope）；评测时需 TaskConfig dataset_hub=huggingface
        dataset_id='MathArena/hmmt_nov_2025',
        tags=[Tags.MATH, Tags.REASONING],
        description='Custom HMMT benchmark adapter for MathArena hmmt_nov_2025.',
        subset_list=['default'],
        # HMMT 是正式评测题，不建议 few-shot
        few_shot_num=0,
        # HF 上 MathArena/hmmt_nov_2025 仅有 train split，无 test
        train_split='train',
        eval_split='train',
        metric_list=['acc'],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class HMMTNov25Adapter(DefaultDataAdapter):
    """
    Adapter for HMMT-style final-answer math problems.

    Expected fields in each record:
      - problem_idx: int
      - problem: str
      - answer: str
      - problem_type: list[str] or str
    """

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record['problem']).strip()
        target = self._normalize_answer(str(record['answer']))
        qid = record.get('problem_idx')
        problem_type = record.get('problem_type', [])

        # problem_type 可能是 list，也可能是字符串
        if isinstance(problem_type, str):
            subset_key = problem_type.strip().lower() or 'default'
        elif isinstance(problem_type, list) and len(problem_type) > 0:
            # 先取第一个 type 作为 subset；后续再精细化
            subset_key = str(problem_type[0]).strip().lower()
        else:
            subset_key = 'default'

        return Sample(
            input=problem,
            target=target,
            subset_key=subset_key,
            metadata={
                'problem_idx': qid,
                'problem_type': problem_type,
                'raw_answer': str(record['answer']),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState):
        """
        从模型输出中抽最终答案。
        优先匹配：
          ANSWER: xxx
        否则回退到最后一行。
        """
        if prediction is None:
            return ''

        text = prediction.strip()

        # 优先抽 ANSWER: ...
        m = re.search(r'ANSWER\s*:\s*(.+)', text, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            return self._normalize_answer(ans)

        # 回退：取最后一行
        last_line = text.splitlines()[-1].strip() if text.splitlines() else text
        return self._normalize_answer(last_line)

    @staticmethod
    def _normalize_answer(ans: str) -> str:
        """
        第一版只做“足够稳”的轻量规范化：
        - 去掉首尾空白
        - 去掉包裹性的 $...$
        - 去掉 \\boxed{...}
        - 合并空白
        - 去掉末尾句号
        """
        if ans is None:
            return ''

        s = str(ans).strip()

        # 去 $...$
        if len(s) >= 2 and s[0] == '$' and s[-1] == '$':
            s = s[1:-1].strip()

        # 去 \boxed{...}
        boxed_match = re.fullmatch(r'\\boxed\{(.+)\}', s)
        if boxed_match:
            s = boxed_match.group(1).strip()

        # 去掉多余空白
        s = re.sub(r'\s+', ' ', s).strip()

        # 去掉末尾句号
        s = s.rstrip('.').strip()

        return s
