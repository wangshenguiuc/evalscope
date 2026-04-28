import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='kormedmcqa',
        pretty_name='KorMedMCQA',
        dataset_id='sean0042/KorMedMCQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        subset_list=['dentist', 'doctor', 'nurse', 'pharm'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class KorMedMCQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['A'], record['B'], record['C'],
                   record['D'], record['E']]
        target = string.ascii_uppercase[int(record['answer']) - 1]
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
            metadata={
                'subject': record.get('subject') or self.current_subset_name,
                'year': record.get('year'),
            },
        )
