from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_OPTION_KEYS = ('option_a', 'option_b', 'option_c', 'option_d', 'option_e')


@register_benchmark(
    BenchmarkMeta(
        name='m_arc',
        pretty_name='m_arc',
        dataset_id='alexandrainst/m_arc',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        subset_list=[
            'ar', 'bn', 'ca', 'da', 'de', 'en', 'es', 'eu', 'fr', 'gu', 'hi',
            'hr', 'hu', 'hy', 'id', 'is', 'it', 'kn', 'ml', 'mr', 'nb', 'ne',
            'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi',
            'zh',
        ],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MArcAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = []
        for key in _OPTION_KEYS:
            choice = record.get(key)
            if choice is None or choice == '':
                continue
            choices.append(choice)
        return Sample(
            input=record['instruction'],
            choices=choices,
            target=record['answer'],
            metadata={
                'id': record.get('id'),
                'language': self.current_subset_name,
            },
        )
