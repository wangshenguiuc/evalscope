"""
MultiChallenge Official：与官方 multi-challenge 仓库完全对齐的评估。

与现有 ``multichallenge`` adapter 的区别：
- **结构化输出**：直接用 OpenAI ``beta.chat.completions.parse(response_format=JudgeResponse)``
  获取结构化的 ``{reasoning, verdict}``，与官方仓库一致，不走正则提取。
- **分数聚合**：按 4 个 AXIS 分别算通过率，再求 **macro-average**（官方算法），
  而非 273 题直接 mean（micro-average）。

官方 AXIS（4 个维度）：
  INFERENCE_MEMORY, INSTRUCTION_RETENTION, RELIABLE_VERSION_EDITING, SELF_COHERENCE
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.metric.scorer import AggScore, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


class JudgeResponse(BaseModel):
    """与官方 multi-challenge/src/evaluator.py 的 JudgeResponse 一致。"""
    reasoning: str
    verdict: Literal["YES", "NO"]


# 与 multi-challenge/src/evaluator.py 一致
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
        name="multichallenge_official",
        pretty_name="MultiChallenge (Official)",
        dataset_id=os.environ.get("MULTICHALLENGE_HF_ID", "nmayorga7/multichallenge"),
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING],
        description="""
## MultiChallenge (Official)

与官方 multi-challenge 仓库对齐的评估：按 4 个 AXIS **macro-average** 计算总分。

数据同 ``multichallenge``；裁判 prompt 一致；唯一区别是聚合方式。
""",
        subset_list=["default"],
        few_shot_num=0,
        train_split=None,
        eval_split=os.environ.get("MULTICHALLENGE_HF_SPLIT", "train"),
        metric_list=["acc"],
        prompt_template="{question}",
    )
)
class MultichallengeOfficialAdapter(DefaultDataAdapter):
    """MultiChallenge Official：macro-average by AXIS。"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True
        self.reformat_subset = True
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

    def _get_openai_client(self) -> OpenAI:
        """从 llm_judge 上取 api_key/api_url/model_id，构建独立的 OpenAI client。"""
        if not hasattr(self, "_openai_client"):
            if self.llm_judge is None:
                raise RuntimeError("llm_judge is None; check judge_model_args and judge_strategy.")
            self._openai_client = OpenAI(
                api_key=self.llm_judge.api_key,
                base_url=self.llm_judge.api_url,
            )
            self._judge_model_id = "gpt-4o-2024-08-06"
        return self._openai_client

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
            logger.error("MultichallengeOfficial: llm_judge is None; check judge_model_args and judge_strategy.")
            score.value = {"acc": 0.0}
            score.explanation = "LLM judge not initialized"
            score.main_score_name = "acc"
            return score

        # 直接调 OpenAI 结构化输出，与官方仓库一致
        try:
            client = self._get_openai_client()
            response = client.beta.chat.completions.parse(
                model=self._judge_model_id,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                response_format=JudgeResponse,
            )
            judge_result = response.choices[0].message.parsed
            verdict = judge_result.verdict
            reasoning = judge_result.reasoning
        except Exception as e:
            logger.error(f"MultichallengeOfficial structured output failed: {e}")
            verdict = ""
            reasoning = f"[ERROR] {e}"

        acc = 1.0 if verdict == pass_criteria else 0.0
        score.value = {"acc": acc}
        score.explanation = f"LLM judge (MultichallengeOfficial): reasoning={reasoning}, verdict={verdict}"
        score.metadata = {
            "source": "llm_judge_structured",
            "model": getattr(self, "_judge_model_id", None),
            "verdict": verdict,
            "pass_criteria": pass_criteria,
            "question_id": meta.get("question_id"),
            "axis": meta.get("axis"),
        }
        score.main_score_name = "acc"
        return score

    # ------------------------------------------------------------------
    # 关键区别：按 AXIS macro-average，与官方 multi-challenge 仓库一致
    # ------------------------------------------------------------------
    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        官方算法：
        1. 按 AXIS 分组
        2. 每个 AXIS 内算 acc 的平均值（通过率）
        3. 总分 = 各 AXIS 通过率的平均值（macro-average）
        """
        # 按 axis 分组
        axis_scores: Dict[str, List[float]] = defaultdict(list)
        for ss in sample_scores:
            axis = (ss.sample_metadata or {}).get("axis", "default")
            acc = ss.score.value.get("acc", 0.0)
            axis_scores[axis].append(acc)

        # 每个 axis 的通过率
        axis_means: Dict[str, float] = {}
        agg_list: List[AggScore] = []
        for axis, scores in sorted(axis_scores.items()):
            mean_val = sum(scores) / len(scores) if scores else 0.0
            axis_means[axis] = mean_val
            agg_list.append(AggScore(
                metric_name=f"acc__{axis}",
                score=mean_val,
                num=len(scores),
                metadata={"axis": axis},
            ))

        # macro-average：各 axis 等权平均
        macro_avg = sum(axis_means.values()) / len(axis_means) if axis_means else 0.0
        agg_list.insert(0, AggScore(
            metric_name="acc",
            score=macro_avg,
            num=sum(len(v) for v in axis_scores.values()),
            metadata={"aggregation": "macro_average_by_axis", "axis_scores": axis_means},
        ))

        return agg_list
