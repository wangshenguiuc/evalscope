import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# `eu` is the canonical Basque source; `en` is the canonical English
# translation. The other 14 configs (en_mt_*) are model-generated MT
# artifacts the dataset retained for translation-quality research, not
# for capability evaluation — excluded from this wire.
SUBSET_LIST = ['eu', 'en']


@register_benchmark(
    BenchmarkMeta(
        name='bertaqa',
        pretty_name='BertaQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

BertaQA (`HiTZ/BertaQA`) is a Basque cultural / general knowledge
3-choice MCQ benchmark. The dataset ships in `eu` (Basque), `en`
(English), and 14 machine-translation variants (`en_mt_*`); only the
two canonical configs are wired here.

## Task Description

- **Task Type**: 3-choice MCQ
- **Input**: `question` plus `candidates` (list of 3 strings)
- **Output**: Single correct answer letter (A/B/C); `answer` is a
  0-indexed int.

## Evaluation Notes

- 4756 test rows per config; no train/dev splits.
- Primary metric: **Accuracy**.
""",
        dataset_id='HiTZ/BertaQA',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class BertaQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = list(record['candidates'])
        target = string.ascii_uppercase[int(record['answer'])]
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
            metadata={
                'id': record.get('id'),
                'category': record.get('category'),
                'group': record.get('group'),
                'difficulty': record.get('difficulty'),
                'language': self.current_subset_name,
            },
        )
