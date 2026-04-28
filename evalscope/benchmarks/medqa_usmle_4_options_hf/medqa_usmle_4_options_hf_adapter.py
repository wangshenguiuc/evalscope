from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_INT_TO_LETTER = ['A', 'B', 'C', 'D']


@register_benchmark(
    BenchmarkMeta(
        name='medqa_usmle_4_options_hf',
        pretty_name='MedQA-USMLE-4-options-hf',
        dataset_id='GBaker/MedQA-USMLE-4-options-hf',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MedQAUSMLE4OptionsHFAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record['ending0'], record['ending1'],
                   record['ending2'], record['ending3']]
        target = _INT_TO_LETTER[int(record['label'])]
        return Sample(
            input=record['sent1'],
            choices=choices,
            target=target,
            metadata={'id': record.get('id')},
        )
