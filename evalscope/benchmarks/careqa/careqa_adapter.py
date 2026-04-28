import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='careqa',
        pretty_name='CareQA',
        dataset_id='HPAI-BSC/CareQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        # MCQ subsets only; CareQA_en_open is free-form (out of scope for the rule scorer).
        subset_list=['CareQA_en', 'CareQA_es'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class CareQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['op1'], record['op2'], record['op3'], record['op4']]
        target = string.ascii_uppercase[int(record['cop']) - 1]
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
            metadata={
                'category': record.get('category'),
                'year': record.get('year'),
                'unique_id': record.get('unique_id'),
            },
        )
