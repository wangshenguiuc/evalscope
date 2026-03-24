"""
HLE Verified 评测：在 cais/hle 的 test 划分上，仅保留与 **lmms-lab/HLE-Verified** 对齐的题目 id。

与 stock `hle` 的差异：
- 子集 id 来源（默认 **HF**）：**lmms-lab/HLE-Verified · split=test 全量**（约 1811）；若只要 Gold 子集可设 **`HLE_VERIFIED_HF_GOLD_ONLY=1`**。本地 jsonl：`HLE_VERIFIED_SOURCE=jsonl`。
- 裁判：prompt 对齐 hle_eval/run_judge_results.py，末尾仍要求 GRADE: C/I 供 EvalScope 解析。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.hle.hle_adapter import HLEAdapter, SUBSET_LIST
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


def _norm_id(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _env_nonempty(key: str, default: str) -> str:
    """
    读取环境变量；若未设置、或仅为空白，则返回 default。
    注意：bench/run.sh 里 `export VAR="${VAR:-}"` 会把 VAR 设成空字符串，
    此时 os.environ.get(key, default) 仍得到 ''，会导致 load_dataset('', ...) 在 datasets 内 IndexError。
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    s = str(raw).strip()
    return s if s else default


# 与 hle_eval/run_judge_results.py 一致的核心判题说明，并强制最后一行 GRADE 供 EvalScope 使用
HLE_VERIFIED_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Finally, on the last line of your response, output exactly one of: GRADE: C (correct) or GRADE: I (incorrect). This line must match your yes/no judgement above.
"""


def _default_verified_jsonl() -> str:
    root = os.environ.get("EVAL_DATASET_DIR", "/mnt/STEM/data/benchmarks")
    return os.path.join(root, "hle_verified", "Gold_subset.jsonl")


def _load_verified_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("id")
            if qid is not None:
                ids.add(_norm_id(qid))
    return ids


def _row_question_id(row: Any) -> Any:
    """HF 行可能是 dict 或支持 [] 的映射；id 列名因数据集而异。"""
    if row is None:
        return None
    get = row.get if hasattr(row, "get") else None
    for key in ("id", "hle_id", "question_id", "uuid", "problem_id"):
        try:
            if get:
                v = get(key)
            else:
                v = row[key]
        except (KeyError, TypeError, IndexError):
            v = None
        if v is not None and v != "":
            return v
    return None


def _with_hf_token(base: dict) -> dict:
    """与 mirobench_dev/scripts/check_hle_verified_overlap.py 一致：gated 集需 token 或本机 huggingface-cli 缓存。"""
    out = dict(base)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        out["token"] = tok
    else:
        out["token"] = True
    return out


def _row_passes_gold_filter(row: Any, gold_only: bool, gold_tag: str) -> bool:
    """
    新版 HF 仅提供 test split，用 subset / subset_raw 区分 gold 与 revision 等。
    旧版独立 gold split 可无 subset 列，此时 gold_only 下仍保留行。
    """
    if not gold_only:
        return True
    sub = row.get("subset") if hasattr(row, "get") else None
    if sub is not None and str(sub).strip() != "":
        return str(sub).strip().lower() == gold_tag.strip().lower()
    raw = str(row.get("subset_raw", "") if hasattr(row, "get") else "").lower()
    if not raw:
        return True  # 无标签时视为 gold-only 单 split（兼容旧数据）
    return gold_tag.lower() in raw and "revision" not in raw


def _load_verified_ids_from_hf() -> Set[str]:
    """从 HuggingFace 读取 id：默认 **test 全量**；`HLE_VERIFIED_HF_GOLD_ONLY=1` 时只保留 subset=gold。"""
    from datasets import DatasetDict, load_dataset

    hf_id = _env_nonempty("HLE_VERIFIED_HF_ID", "lmms-lab/HLE-Verified")
    split = _env_nonempty("HLE_VERIFIED_HF_SPLIT", "test")
    revision = os.environ.get("HLE_VERIFIED_HF_REVISION", "").strip() or None
    gold_only = os.environ.get("HLE_VERIFIED_HF_GOLD_ONLY", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    gold_tag = os.environ.get("HLE_VERIFIED_HF_GOLD_TAG", "gold").strip() or "gold"

    kwargs = _with_hf_token({"trust_remote_code": True})
    if revision:
        kwargs["revision"] = revision

    try:
        ds = load_dataset(hf_id, split=split, **kwargs)
    except Exception as e:
        err = str(e).lower()
        if "unknown split" in err and split in ("gold", "full", "revision"):
            split = "test"
            ds = load_dataset(hf_id, split=split, **kwargs)
        else:
            raise

    # 未传 split 时可能得到 DatasetDict；与 check 脚本一致按 split 取列
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise RuntimeError(
                f"HLE Verified: HF 返回 DatasetDict，但 split={split!r} 不在 {list(ds.keys())} 中。"
                " 请设置 HLE_VERIFIED_HF_SPLIT 为可用划分（一般为 test）。"
            )
        ds = ds[split]

    ids: Set[str] = set()
    # 用迭代而非 ds[i]：部分 HF datasets / 缓存下 len 与 __getitem__ 可能不一致，会触发
    # IndexError: list index out of range（见 GitHub datasets 相关 issue）。
    for row in ds:
        if not _row_passes_gold_filter(row, gold_only, gold_tag):
            continue
        qid = _row_question_id(row)
        if qid is not None:
            ids.add(_norm_id(qid))
    return ids


# 默认与 bench/run.sh 一致：HF lmms-lab/HLE-Verified · test 全量 id
_DEFAULT_VERIFIED_SOURCE = "hf"


def _load_cais_hle_test_ids() -> Set[str]:
    """与 check_hle_verified_overlap.py 一致：cais/hle · split=test 的题目 id（gated 需 token）。"""
    from datasets import DatasetDict, load_dataset

    split = os.environ.get("HLE_CAIS_HLE_SPLIT", "test").strip() or "test"
    kwargs = _with_hf_token({"trust_remote_code": True})
    try:
        ds = load_dataset("cais/hle", split=split, **kwargs)
    except Exception as e:
        raise RuntimeError(
            "HLE Verified: 无法加载 cais/hle 以校验交集（需 Hub 接受条款 + HF_TOKEN 或 huggingface-cli login）。\n"
            f"  原始错误: {e}"
        ) from e

    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise RuntimeError(
                f"HLE Verified: cais/hle 返回 DatasetDict，但 split={split!r} 不在 {list(ds.keys())} 中。"
            )
        ds = ds[split]

    ids: Set[str] = set()
    for row in ds:
        qid = row.get("id") if hasattr(row, "get") else None
        if qid is None:
            try:
                qid = row["id"]
            except (KeyError, TypeError, IndexError):
                qid = None
        if qid is not None and _norm_id(qid):
            ids.add(_norm_id(qid))
    return ids


def _check_and_log_intersection(verified_ids: Set[str], provenance: str) -> None:
    """
    先统计 Verified id 与 cais/hle test 的交集并打日志（与 mirobench_dev/scripts/check_hle_verified_overlap.py 一致）。
    设 HLE_VERIFIED_SKIP_INTERSECT_CHECK=1 可跳过（不推荐）。
    """
    skip = os.environ.get("HLE_VERIFIED_SKIP_INTERSECT_CHECK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if skip:
        logger.warning(
            "HLE Verified: 已跳过交集校验（HLE_VERIFIED_SKIP_INTERSECT_CHECK=1），"
            "若评测异常请取消该环境变量查看 |Verified|∩|cais/hle test|。"
        )
        return

    hle_ids = _load_cais_hle_test_ids()
    inter = verified_ids & hle_ids
    n_v, n_h, n_i = len(verified_ids), len(hle_ids), len(inter)
    only_v = len(verified_ids - hle_ids)

    # 主日志：先输出交集规模（用户最关心）
    logger.info(
        "HLE Verified 交集: |Verified|=%d |cais/hle test|=%d |intersect|=%d |仅在Verified不在cais|=%d （来源: %s）",
        n_v,
        n_h,
        n_i,
        only_v,
        provenance,
    )
    print(
        f"[hle_verified] intersect: |Verified|={n_v} |cais/hle test|={n_h} "
        f"|intersect|={n_i} |only_in_Verified|={only_v}  (provenance: {provenance})",
        flush=True,
    )

    if n_i == 0:
        raise RuntimeError(
            f"HLE Verified: Verified 与 cais/hle test 交集为 0（|Verified|={n_v}, |cais/hle test|={n_h}）。\n"
            "  请检查：HF 是否可访问、id 是否与 cais/hle 一致、或尝试对齐 id 格式（strip）。\n"
            f"  id 来源: {provenance}"
        )


def _resolve_verified_ids() -> tuple[Set[str], str]:
    """
    Returns:
        (id set, provenance string for error messages)
    """
    raw = os.environ.get("HLE_VERIFIED_SOURCE", _DEFAULT_VERIFIED_SOURCE)
    source = (raw or _DEFAULT_VERIFIED_SOURCE).strip().lower()
    if source in ("hf", "huggingface", "hub"):
        hf_id = _env_nonempty("HLE_VERIFIED_HF_ID", "lmms-lab/HLE-Verified")
        split = _env_nonempty("HLE_VERIFIED_HF_SPLIT", "test")
        gonly = os.environ.get("HLE_VERIFIED_HF_GOLD_ONLY", "0").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        tag = os.environ.get("HLE_VERIFIED_HF_GOLD_TAG", "gold").strip() or "gold"
        ids = _load_verified_ids_from_hf()
        extra = f" subset={tag}" if gonly else " (test 全量)"
        return ids, f"HF {hf_id} split={split}{extra}"
    if source not in ("jsonl", "file"):
        raise ValueError(
            "HLE_VERIFIED_SOURCE 应为 jsonl 或 hf，当前: "
            + repr(source)
        )
    path = os.environ.get("HLE_VERIFIED_JSONL", "").strip() or _default_verified_jsonl()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "HLE Verified（jsonl 模式）需要 Gold_subset.jsonl。\n"
            "  请设置 HLE_VERIFIED_JSONL=/path/to/Gold_subset.jsonl\n"
            f"  或放到默认路径: {path}\n"
            "  默认模式为 HF（HLE_VERIFIED_SOURCE=hf），若要改用本地 jsonl 请显式：\n"
            "    export HLE_VERIFIED_SOURCE=jsonl"
        )
    ids = _load_verified_ids(path)
    return ids, path


@register_benchmark(
    BenchmarkMeta(
        name="hle_verified",
        pretty_name="HLE Verified (HF test ids)",
        tags=[Tags.KNOWLEDGE, Tags.QA],
        description="""
## HLE Verified

在 **cais/hle** 的 test 划分上，仅评测与 **lmms-lab/HLE-Verified** 对齐的题目（默认 **test 全量 id**）。

- **id 来源（默认 HF）**：**split=test** 全部行；只要 Gold 时设 **`HLE_VERIFIED_HF_GOLD_ONLY=1`**。本地 jsonl：`HLE_VERIFIED_SOURCE=jsonl`。
- 纯文本模型可将 **`include_multi_modal`** 设为 `False`。
- 指标为 **acc**，**LLM judge**（输出 `GRADE: C/I`）。
""",
        dataset_id="cais/hle",
        subset_list=SUBSET_LIST,
        metric_list=["acc"],
        eval_split="test",
        prompt_template="{question}",
        extra_params={
            "include_multi_modal": {
                "type": "bool",
                "description": "Include multi-modal (image) questions during evaluation.",
                "value": True,
            }
        },
    )
)
class HLEVerifiedAdapter(HLEAdapter):
    """
    HLE Verified：父类为 HLEAdapter（多模态、题型分 Exact/MC、LLM judge）。
    通过 sample_filter 仅保留 HF Verified（默认 test 全量）或 jsonl 中的题目 id。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._verified_ids, self._verified_provenance = _resolve_verified_ids()
        if not self._verified_ids:
            raise ValueError(f"HLE Verified: 未读到任何题目 id（来源: {self._verified_provenance}）")
        _check_and_log_intersection(self._verified_ids, self._verified_provenance)

    def load_dataset(self) -> DatasetDict:
        """加载 cais/hle 并过滤；若与 Verified id 无交集则直接报错（避免空 metrics 触发 Report 越界）。"""
        result = super().load_dataset()
        total = 0
        for _k, ds in result.items():
            total += len(ds.samples)
        if total == 0:
            raise RuntimeError(
                "HLE Verified: 过滤后样本数为 0（Verified 题目 id 与 cais/hle test 无交集）。\n"
                "  请确认 HF 数据集与 cais/hle 使用同一套题目 id（默认 lmms-lab/HLE-Verified · test 全量）。\n"
                "  若使用本地列表：检查 HLE_VERIFIED_JSONL 中的 id 是否与 cais/hle 一致。\n"
                "  纯文本评测可尝试 dataset_args: hle_verified.include_multi_modal=false。"
            )
        return result

    def sample_filter(self, sample) -> bool:
        if not super().sample_filter(sample):
            return False
        if sample.metadata is None:
            return False
        uid = sample.metadata.get("uid")
        return _norm_id(uid) in self._verified_ids

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """使用与 hle_eval 对齐的裁判 prompt，保留 GRADE: C/I 解析逻辑。"""
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        confidence = 100
        if task_state.output and task_state.output.completion:
            confidence_match = re.search(r"confidence:\s*(\d+)", task_state.output.completion, re.IGNORECASE)
            if confidence_match:
                confidence = int(confidence_match.group(1))

        judge_prompt = HLE_VERIFIED_JUDGE_PROMPT.format(
            question=task_state.input_text,
            response=filtered_prediction,
            correct_answer=reference,
        )

        judge_response = self.llm_judge.judge(prompt=judge_prompt)

        accuracy_score = re.search(r"GRADE:\s*([CI])", judge_response, re.IGNORECASE)
        if accuracy_score:
            grade = accuracy_score.group(1).upper()
            score.value = {
                "acc": 1.0 if grade == "C" else 0.0,
            }
        else:
            score.value = {
                "acc": 0.0,
            }
        score.explanation = f"LLM judge (HLE Verified / hle_eval-style): {judge_response}"
        score.metadata = {
            "source": "llm_judge",
            "judge_strategy": self.judge_strategy,
            "model": self.llm_judge.model_id,
            "confidence": confidence,
        }
        score.main_score_name = "acc"
        return score
