import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='arc_challenge',
        pretty_name='ARC-Challenge',
        dataset_id='ibragim-bad/arc_challenge',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ArcChallengeAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = record['choices']['text']
        answer_key = record['answerKey']
        if answer_key.isdigit():
            answer_key = string.ascii_uppercase[int(answer_key) - 1]
        return Sample(
            input=record['question'],
            choices=choices,
            target=answer_key,
            metadata={'id': record.get('id', '')},
        )
