import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_LETTERS = list('ABCDEFGHIJ')


@register_benchmark(
    BenchmarkMeta(
        name='medxpertqa',
        pretty_name='MedXpertQA',
        dataset_id='TsinghuaC3I/MedXpertQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        # text-only subset; MM (multimodal) excluded — out of scope for qwen3.5-35b-a3b smoke
        subset_list=['Text'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MedXpertQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        question = re.split(r'\nAnswer Choices:', question, maxsplit=1)[0].rstrip()

        options = record['options']
        choices = []
        for letter in _LETTERS:
            choice = options.get(letter, '')
            if not choice:
                break
            choices.append(choice)

        return Sample(
            input=question,
            choices=choices,
            target=record['label'],
            metadata={
                'id': record.get('id'),
                'medical_task': record.get('medical_task'),
                'body_system': record.get('body_system'),
                'question_type': record.get('question_type'),
            },
        )
