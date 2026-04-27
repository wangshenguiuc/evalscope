import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.messages import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    Content,
    ContentImage,
    ContentText,
)
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.hle.hle_adapter import (
    ANSWER_TYPE_EXACT_MATCH,
    HLEAdapter,
    SUBSET_LIST,
    SYSTEM_EXACT_ANSWER,
    SYSTEM_MC,
)
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='hle_verified_skylenage',
        pretty_name='HLE-Verified (skylenage audit)',
        tags=[Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

HLE-Verified (skylenage-ai/HLE-Verified) is a community-driven systematic
audit of CAIS' Humanity's Last Exam (cais/hle), separately published from
the existing `hle_verified` benchmark — which only filters cais/hle to
the lmms-lab Verified id list and reuses the original questions/answers.

This audit instead provides revised questions, revised answers, three
quality tiers (Gold / Revision / Uncertain), and per-item verification
metadata. Source: arXiv:2602.13964 (Feb 2026).

## Task Description

- **Task Type**: Hard, multi-domain knowledge / reasoning QA
- **Input**: revised `question` (Korean / English / etc.), optional image
- **Output**: short answer; `exactMatch` or `multipleChoice`
- **Grading**: HLE-style LLM judge (`GRADE: C` / `GRADE: I`)

## Subsets

The dataset ships a single `default` config with all 2,500 rows; the
three quality tiers are available as the `Verified_Classes` column
(Gold subset = 668, Revision subset = 1,143, Uncertain subset = 689).
Per the dataset card, Gold is recommended for leaderboard-level use,
Revision for robustness sensitivity, and Uncertain for ambiguity research.

## Modality

About 13.7% of rows carry an image inside the embedded `json` blob
(rate is roughly uniform across the three Verified_Classes tiers). To
keep the wire text-only by default, set
`extra_params.include_multi_modal=False` — the parent `HLEAdapter`'s
`sample_filter` then drops image-bearing samples. Vision-capable models
should set it to `True`.
""",
        dataset_id='skylenage-ai/HLE-Verified',
        # Same 8 categories as HLE (Biology/Medicine, Chemistry, ..., Physics, Other).
        # Required for reformat_subset=True (inherited from HLEAdapter): the parent
        # loads the upstream `default` config, re-keys samples by their
        # subset_key=record['category'], and filters by this subset_list.
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # the only split that ships
        prompt_template='{question}',
        extra_params={
            'include_multi_modal': {
                'type': 'bool',
                'description': 'Include multi-modal (image) questions during evaluation.',
                'value': True,
            },
        },
    )
)
class HLEVerifiedSkylenageAdapter(HLEAdapter):
    """skylenage's audited HLE variant.

    Upstream layout differs from cais/hle: top-level columns are
    `id`/`question`/`answer`/`category`/`raw_subject`/`Verified_Classes`
    plus a stringified `json` blob holding the original HLE-style fields
    (`image`, `answer_type`, `rationale`, `author_name`, ...). This
    adapter parses `json` to recover the multi-modal payload and the
    answer-type-driven system prompt selection, then delegates the
    `sample_filter` and `llm_match_score` paths to `HLEAdapter` unchanged.
    """

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        try:
            jd = json.loads(record.get('json') or '{}')
        except (ValueError, TypeError):
            jd = {}

        answer_type = jd.get('answer_type') or ANSWER_TYPE_EXACT_MATCH
        system_prompt = (
            SYSTEM_EXACT_ANSWER if answer_type == ANSWER_TYPE_EXACT_MATCH else SYSTEM_MC
        )
        image = jd.get('image') or ''

        text_content = ContentText(text=record['question'])
        content: List[Content] = [text_content]
        if image:
            content.append(ContentImage(image=image))

        messages: List[ChatMessage] = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=content),
        ]
        return Sample(
            input=messages,
            subset_key=record.get('category'),
            metadata={
                'uid': record['id'],
                'author_name': jd.get('author_name'),
                'rationale': jd.get('rationale'),
                'raw_subject': record.get('raw_subject'),
                'category': record.get('category'),
                'has_image': bool(image),
                'verified_classes': record.get('Verified_Classes'),
                'answer_type': answer_type,
                'problem_is_valid': record.get('problem_is_valid'),
                'answer_is_valid': record.get('answer_is_valid'),
                'rationale_is_valid': record.get('rationale_is_valid'),
            },
            target=record['answer'],
        )
