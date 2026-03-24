"""
MultiChallenge：多轮对话 + LLM Judge（YES/NO），与官方 `multi-challenge/src/evaluator.py` 对齐。

- 数据：默认 HF `nmayorga7/multichallenge`（`eval_split` 默认 `train`，可用 `dataset_args` 覆盖）；
  若 Hub 不可用，请设置 **`MULTICHALLENGE_JSONL`** 指向官方 `benchmark_questions.jsonl`，
  或在 `dataset_args.multichallenge` 中传 **`data_path`**。
- 判分：`PASS_CRITERIA`（期望 verdict，如 YES/NO）与 judge 输出 verdict 是否一致；指标 **acc**。
- 裁判：走 EvalScope **`judge_model_args`**（与 mirobench `bench/run_eval.py` 的 `--judge-*` 一致），非 OpenAI 硬编码。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# 与 multi-challenge/src/evaluator.py 一致（第二段为 TARGET_QUESTION）
JUDGE_PROMPT = """You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{model_response}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{target_question}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO"."""


def _pick(record: Dict[str, Any], *candidates: str) -> Any:
    for k in candidates:
        if k in record and record[k] is not None:
            return record[k]
    return None


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif "text" in p:
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def _conversation_to_messages(conversation: Any) -> List[ChatMessage]:
    if not isinstance(conversation, list):
        raise ValueError("CONVERSATION must be a list of message dicts")
    out: List[ChatMessage] = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).lower().strip()
        text = _content_to_str(msg.get("content"))
        if role == "system":
            out.append(ChatMessageSystem(content=text))
        elif role == "assistant":
            out.append(ChatMessageAssistant(content=text))
        else:
            out.append(ChatMessageUser(content=text))
    if not out:
        raise ValueError("Empty CONVERSATION after parsing")
    return out


@register_benchmark(
    BenchmarkMeta(
        name="multichallenge",
        pretty_name="MultiChallenge",
        dataset_id=os.environ.get("MULTICHALLENGE_HF_ID", "nmayorga7/multichallenge"),
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING],
        description="""
## MultiChallenge

多轮对话 benchmark；裁判 prompt 与官方仓库 `evaluator.py` 一致，期望 verdict 与数据中的 **PASS_CRITERIA** 一致。

**数据**：默认 HuggingFace `nmayorga7/multichallenge`（`eval_split` 默认 `train`）。若加载失败，请设置环境变量 **`MULTICHALLENGE_JSONL`** 为本地 `benchmark_questions.jsonl`，或在 `dataset_args.multichallenge` 中设置 **`data_path`**。

**评测**：必须配置 **`judge_model_args`**（OpenAI 兼容 API），与 mirobench `run_eval.py` 的 `--judge-api-key` / `--judge-api-url` / `--judge-model-id` 一致。
""",
        subset_list=["default"],
        few_shot_num=0,
        train_split=None,
        # 常见为 train；若 HF 卡与本地数据 split 不同，可设 MULTICHALLENGE_HF_SPLIT 或 dataset_args
        eval_split=os.environ.get("MULTICHALLENGE_HF_SPLIT", "train"),
        metric_list=["acc"],
        prompt_template="{question}",
    )
)
class MultichallengeAdapter(DefaultDataAdapter):
    """MultiChallenge：对话输入 + LLM judge（YES/NO vs PASS_CRITERIA）。"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True
        self.reformat_subset = True
        # 单桶 default，避免 AXIS 与预设列表不一致时样本被静默丢弃（AXIS 见 metadata）
        self.subset_list = ["default"]

    def _dataset_extra(self) -> Dict[str, Any]:
        if not self._task_config or not self._task_config.dataset_args:
            return {}
        return self._task_config.dataset_args.get(self.name, {}) or {}

    def _resolve_local_jsonl_path(self) -> Optional[str]:
        extra = self._dataset_extra()
        for key in ("data_path", "jsonl_path", "path"):
            p = extra.get(key)
            if p and os.path.isfile(str(p)):
                return str(p)
        env = os.environ.get("MULTICHALLENGE_JSONL", "").strip()
        if env and os.path.isfile(env):
            return env
        root = os.environ.get("MULTICHALLENGE_ROOT", "").strip()
        if root:
            candidate = os.path.join(root, "data", "benchmark_questions.jsonl")
            if os.path.isfile(candidate):
                return candidate
        return None

    def load(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        p = self._resolve_local_jsonl_path()
        if p:
            old_id = self._benchmark_meta.dataset_id
            self._benchmark_meta.dataset_id = p
            try:
                with self._temporary_attribute("dataset_hub", HubType.LOCAL):
                    return self.load_from_disk(use_local_loader=True)
            finally:
                self._benchmark_meta.dataset_id = old_id
        return super().load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        qid = _pick(record, "QUESTION_ID", "question_id", "id")
        axis = str(_pick(record, "AXIS", "axis") or "default").strip() or "default"
        conv = _pick(record, "CONVERSATION", "conversation")
        target_q = str(_pick(record, "TARGET_QUESTION", "target_question") or "").strip()
        pass_crit = _pick(record, "PASS_CRITERIA", "pass_criteria")
        if pass_crit is None:
            raise ValueError("record missing PASS_CRITERIA / pass_criteria")
        pass_s = str(pass_crit).strip().upper()

        messages = _conversation_to_messages(conv)

        return Sample(
            input=messages,
            target=pass_s,
            subset_key="default",
            metadata={
                "question_id": qid,
                "axis": axis,
                "target_question": target_q,
                "pass_criteria": pass_s,
            },
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        meta = task_state.metadata or {}
        target_question = str(meta.get("target_question") or "").strip()
        pass_criteria = str(meta.get("pass_criteria") or reference or "").strip().upper()

        judge_prompt = JUDGE_PROMPT.format(
            model_response=filtered_prediction or "",
            target_question=target_question,
        )

        if self.llm_judge is None:
            logger.error("MultiChallenge: llm_judge is None; check judge_model_args and judge_strategy.")
            score.value = {"acc": 0.0}
            score.explanation = "LLM judge not initialized"
            score.main_score_name = "acc"
            return score

        judge_response = self.llm_judge.judge(prompt=judge_prompt)

        verdict_m = re.findall(r"\b(YES|NO)\b", judge_response, flags=re.IGNORECASE)
        verdict = verdict_m[-1].upper() if verdict_m else ""

        acc = 1.0 if verdict == pass_criteria else 0.0
        score.value = {"acc": acc}
        score.explanation = f"LLM judge (MultiChallenge): {judge_response}"
        score.metadata = {
            "source": "llm_judge",
            "judge_strategy": self.judge_strategy,
            "model": getattr(self.llm_judge, "model_id", None),
            "verdict": verdict,
            "pass_criteria": pass_criteria,
            "question_id": meta.get("question_id"),
            "axis": meta.get("axis"),
        }
        score.main_score_name = "acc"
        return score
