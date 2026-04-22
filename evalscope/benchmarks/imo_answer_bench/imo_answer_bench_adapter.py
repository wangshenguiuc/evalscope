from __future__ import annotations

import os
from typing import Any, Dict
import re

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Prompt format selector.
#   "answer"        — baseline: final line "ANSWER: <final_answer>".
#   "boxed"         — per Qwen3.5/3.6 model card recommendation: \boxed{}.
#   "answer_verify" — ANSWER: format + explicit "identify question type + verify
#                     before committing" guidance. Targets failure modes where the
#                     model misreads what's being asked or commits without checking.
_VALID_FORMATS = ('answer', 'boxed', 'answer_verify')
_PROMPT_FORMAT = os.environ.get('IMO_PROMPT_FORMAT', 'answer').strip().lower()
if _PROMPT_FORMAT not in _VALID_FORMATS:
    raise ValueError(
        f'IMO_PROMPT_FORMAT must be one of {_VALID_FORMATS}, got {_PROMPT_FORMAT!r}'
    )

# Judge mode selector.
#   "autograder" (default) — paper-aligned (arxiv 2511.01846 §A.5): pass tail of
#                            full model solution, parse `\boxed{Correct}`/`\boxed{Incorrect}`.
#   "verdict"              — legacy path: pass extracted answer, parse `VERDICT: EQUIVALENT`.
# Autograder is the default because it catches rule-based false positives (e.g. off-by-one
# integers) and correctly handles parametric-family answers; see runs 04180940 / 04191756.
_VALID_JUDGE_MODES = ('verdict', 'autograder')
_JUDGE_MODE = os.environ.get('IMO_JUDGE_MODE', 'autograder').strip().lower()
if _JUDGE_MODE not in _VALID_JUDGE_MODES:
    raise ValueError(
        f'IMO_JUDGE_MODE must be one of {_VALID_JUDGE_MODES}, got {_JUDGE_MODE!r}'
    )

# Max chars of model solution to hand to the autograder (<think> already stripped
# upstream by eval.py's remove_until filter). Tail slice — final answer lives at end.
_AUTOGRADER_SOLUTION_CHARS = int(os.environ.get('IMO_AUTOGRADER_SOLUTION_CHARS', '2000'))

# LLM judge prompt for math equivalence (used as fallback when rule-based grader fails).
# Adapted from GDM answer auto-grader (Section A.5 of https://arxiv.org/abs/2511.01846).
# NOTE: Use double braces {{}} to escape literal braces from str.format().
JUDGE_SYSTEM_PROMPT = """\
# System Role: Deterministic Mathematical Autograder

You are a precise, automated grading system. Your sole function is to determine if the final \
answer provided in the Model Solution is mathematically equivalent to the Golden Answer. \
You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)
Equivalence is mandatory for a correct grade. You must rigorously \
verify if the answers represent the exact same mathematical value \
or expression, even if the format differs.
* **Algebraic Equivalence:** e.g., `n(n+1)/2` is equivalent to `n^2/2 + n/2`. You must verify the algebra.
* **Numerical Equivalence:** e.g., `1/2` is equivalent to `0.5`; `sqrt(2)/2` is equivalent to `1/sqrt(2)`. \
You must use the provided numerical tolerance when comparing approximate decimal forms. \
Treat two numeric answers as equivalent when their absolute difference is less than or equal to the stated tolerance.
* **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter \
(e.g., {{1,2}} is equivalent to {{2, 1}}).
* **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 2. Output Protocol (Strict Compliance Required)
You must execute the task using a two-part structure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. \
Make your thinking concise. This section details your reasoning \
process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. \
Detail the steps taken to verify mathematical equivalence \
(e.g., simplification, algebraic manipulation). \
You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** Decide "Equivalent" or "Not Equivalent".

**Part 2: Final Verdict (Strict Format)**
Immediately after the closing </thinking> tag, output EXACTLY ONE of the following lines \
and NOTHING ELSE — no explanation, no punctuation, no extra text, no trailing whitespace:

VERDICT: EQUIVALENT
VERDICT: NOT_EQUIVALENT

This final line is machine-parsed. Any deviation from the exact format above will be treated as a grading failure.
"""

JUDGE_USER_TEMPLATE = """\
Here is the problem, predicted answer, and golden answer to grade.

Problem: {problem}
Predicted Answer: {pred}
Golden Answer: {gold}
Numerical Tolerance: 1e-6

After your <thinking></thinking> analysis, output exactly one line:
`VERDICT: EQUIVALENT` or `VERDICT: NOT_EQUIVALENT`. No other text.
"""

# ---------------------------------------------------------------------------
# AnswerAutoGrader — adapted from arxiv 2511.01846 §A.5 (IMO-ProofBench).
# Activated by IMO_JUDGE_MODE=autograder. Deviations from verbatim paper text:
#   - Section renumbered 3 -> 2 (paper skips 2 with no section defined).
#   - Dropped "based on the Extraction Protocol" (dangling reference).
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT_AUTOGRADER = r"""# System Role: Deterministic Mathematical Autograder

You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer. You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)

Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.

*   Algebraic Equivalence: e.g., 'n(n+1)/2' is equivalent to 'n^2/2 + n/2'. You must verify the algebra.

*   Numerical Equivalence: e.g., '1/2' is equivalent to '0.5'; 'sqrt(2)/2' is equivalent to '1/sqrt(2)'.

*   Set/List Equivalence: Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {1, 2} is equivalent to {2, 1}).

*   Partial Credit: No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.

*   No Answers: If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 2. Output Protocol (Strict Compliance Required)

You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:

1. **Golden Answer:** State the Golden Answer.

2. **Extracted Model Answer:** State the extracted final answer from the Model Solution. If none found, state "No clear final answer found."

3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.

4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.

- If Correct: \boxed{Correct}

- If Incorrect: \boxed{Incorrect}

**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \boxed{} output.**

Output example:

<thinking>

1. **Golden Answer:** (−∞,−4)∪(−4,∞)

2. **Extracted Model Answer:** ∅ (the empty set)

3. **Equivalence Analysis:**
   The Golden Answer is a non-empty set of real numbers. The Model Answer is the empty set. These two sets are not equivalent. The empty set contains no elements, while the Golden Answer contains an infinite number of elements.

4. **Conclusion:** Incorrect

</thinking>

\boxed{Incorrect}
"""

# Input block — corresponds to §4 in the paper's query prompt, renumbered here
# for continuity with the renumbered output-protocol section (2).
JUDGE_USER_TEMPLATE_AUTOGRADER = """# 3. Input Data
Here is the problem, model solution, and golden answer to grade:

Problem: `{problem}`
Model Solution: `{solution}`
Golden Answer: `{gold}`
"""

# ANSWER format: final line "ANSWER: <final_answer>". Empirically the strongest
# prompt on this benchmark — the forced restatement acts as a reflection pass.
_PROMPT_TEMPLATE_ANSWER = """
Solve the following math competition problem step by step.

The last line of your response must be exactly in the format:
ANSWER: <final_answer>

Do not use \\boxed{{}}.
Do not add extra text after the final answer line.

Problem:
{question}

Reasoning:
""".lstrip()

# Boxed format: per Qwen3.5/3.6 model card recommendation.
_PROMPT_TEMPLATE_BOXED = """
Solve the following math competition problem.

Please reason step by step, and put your final answer within \\boxed{{}}.

Problem:
{question}

Reasoning:
""".lstrip()

# ANSWER format with explicit question-type check + verification step.
# Addresses two recurring failure modes observed in run #2:
#  - Model commits to an answer in a different form than asked
#    (e.g. "n ≥ 2" when the question wants a specific number).
#  - Model commits without sanity-checking against the problem's conditions.
_PROMPT_TEMPLATE_ANSWER_VERIFY = """
Solve the following math competition problem step by step.

Guidelines:
- First identify what the question asks for — a specific numeric value, a set, a formula, or a characterization. Your final answer must match this expected form.
- After deriving a candidate answer, briefly verify it against the problem's conditions before committing.

The last line of your response must be exactly in the format:
ANSWER: <final_answer>

Do not use \\boxed{{}}.
Do not add extra text after the final answer line.

Problem:
{question}

Reasoning:
""".lstrip()

_TEMPLATES = {
    'answer': _PROMPT_TEMPLATE_ANSWER,
    'boxed': _PROMPT_TEMPLATE_BOXED,
    'answer_verify': _PROMPT_TEMPLATE_ANSWER_VERIFY,
}
PROMPT_TEMPLATE = _TEMPLATES[_PROMPT_FORMAT]


@register_benchmark(
    BenchmarkMeta(
        name='imo_answer_bench',
        pretty_name='IMO-AnswerBench',
        dataset_id='OpenEvals/IMO-AnswerBench',
        tags=[Tags.MATH, Tags.REASONING],
        description='IMO-AnswerBench (OpenEvals/HF); short-answer olympiad problems.',
        subset_list=['algebra', 'combinatorics', 'geometry', 'number theory'],
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

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

        # Extract the canonical signal first based on the active prompt format,
        # then fall back to the other form, then last non-empty line.
        def _match_answer_line():
            # Anchor ANSWER: to start-of-line (MULTILINE) so it doesn't match
            # "Final Answer:" mid-reasoning. Take the LAST match because the
            # prompt instructs "the last line of your response must be ANSWER:".
            matches = re.findall(
                r'^\s*ANSWER\s*:\s*(.+?)\s*$',
                text, flags=re.MULTILINE,
            )
            return self._normalize_answer(matches[-1]) if matches else None

        def _match_last_boxed():
            # Brace-depth-aware extraction handles nested braces like \boxed{2^{2024}+1}.
            val = self._extract_last_boxed(text)
            return self._normalize_answer(val) if val is not None else None

        if _PROMPT_FORMAT == 'boxed':
            extractors = [_match_last_boxed, _match_answer_line]
        else:
            extractors = [_match_answer_line, _match_last_boxed]

        for extractor in extractors:
            result = extractor()
            if result is not None:
                return result

        # Fallback: last non-empty line
        last_line = text.splitlines()[-1].strip() if text.splitlines() else text
        return self._normalize_answer(last_line)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from .grader import grade_answer

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        try:
            is_correct = grade_answer(filtered_prediction, reference)
            score.value['acc'] = 1.0 if is_correct else 0.0
        except Exception as e:
            logger.error(f'Error in IMO grading: {e}')
            score.value['acc'] = 0.0
            score.metadata['acc'] = f'grading_error: {str(e)}'
        return score

    def llm_match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        if _JUDGE_MODE == 'autograder':
            return self._llm_match_score_autograder(
                original_prediction, filtered_prediction, reference, task_state,
            )

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        problem = task_state.input_text if task_state.input_text else 'Not provided'
        user_prompt = JUDGE_USER_TEMPLATE.format(
            problem=problem, pred=filtered_prediction, gold=reference,
        )
        judge_response = self.llm_judge.judge(
            prompt=user_prompt, system_prompt=JUDGE_SYSTEM_PROMPT,
        )

        # Parse strict verdict line. If it's missing (API error, malformed output),
        # surface as a grading error rather than silently scoring 0.
        verdict_match = re.search(
            r'^\s*VERDICT:\s*(EQUIVALENT|NOT_EQUIVALENT)\s*$',
            judge_response, re.MULTILINE,
        )
        metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
        }
        if verdict_match:
            is_correct = verdict_match.group(1) == 'EQUIVALENT'
            score.value = {'acc': 1.0 if is_correct else 0.0}
        else:
            logger.warning(
                f'IMO judge: no VERDICT line found in response (first 200 chars): '
                f'{judge_response[:200]!r}'
            )
            score.value = {'acc': 0.0}
            metadata['grading_error'] = 'missing_verdict_line'

        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = metadata
        score.main_score_name = 'acc'
        return score

    def _llm_match_score_autograder(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        problem = task_state.input_text if task_state.input_text else 'Not provided'
        solution = (original_prediction or '').strip()
        if len(solution) > _AUTOGRADER_SOLUTION_CHARS:
            solution = solution[-_AUTOGRADER_SOLUTION_CHARS:]

        user_prompt = JUDGE_USER_TEMPLATE_AUTOGRADER.format(
            problem=problem, solution=solution, gold=reference,
        )
        judge_response = self.llm_judge.judge(
            prompt=user_prompt, system_prompt=JUDGE_SYSTEM_PROMPT_AUTOGRADER,
        )

        # Parse last \boxed{Correct|Incorrect} after the </thinking> tag.
        verdict_match = None
        for m in re.finditer(r'\\boxed\{\s*(Correct|Incorrect)\s*\}', judge_response):
            verdict_match = m

        metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'judge_mode': 'autograder',
            'model': self.llm_judge.model_id,
            'solution_chars_sent': len(solution),
        }
        if verdict_match:
            is_correct = verdict_match.group(1) == 'Correct'
            score.value = {'acc': 1.0 if is_correct else 0.0}
        else:
            logger.warning(
                f'IMO autograder: no \\boxed{{Correct|Incorrect}} in response '
                f'(first 200 chars): {judge_response[:200]!r}'
            )
            score.value = {'acc': 0.0}
            metadata['grading_error'] = 'missing_boxed_verdict'

        score.explanation = f'LLM autograder: {judge_response}'
        score.metadata = metadata
        score.main_score_name = 'acc'
        return score

    @staticmethod
    def _extract_last_boxed(text: str):
        """Extract content from the last \\boxed{...}, handling nested braces."""
        # Find all \boxed{ positions
        positions = [m.start() for m in re.finditer(r'\\boxed\{', text)]
        if not positions:
            return None
        # Take the last one and extract with brace matching
        start = positions[-1] + len('\\boxed{')
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
        return None

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
