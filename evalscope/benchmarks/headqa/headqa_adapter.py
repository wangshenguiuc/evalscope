import string
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='headqa',
        pretty_name='HEAD-QA',
        dataset_id='EleutherAI/headqa',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        subset_list=['en', 'es'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class HEADQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        answers = record.get('answers') or []
        if not answers:
            return []
        # Sort by aid in case of disordered list; aids are 1-indexed.
        answers_sorted = sorted(answers, key=lambda a: a['aid'])
        choices = [a['atext'] for a in answers_sorted]
        ra = int(record['ra'])
        if ra < 1 or ra > len(choices):
            return []
        target = string.ascii_uppercase[ra - 1]
        return Sample(
            input=record['qtext'],
            choices=choices,
            target=target,
            metadata={
                'category': record.get('category'),
                'qid': record.get('qid'),
            },
        )
