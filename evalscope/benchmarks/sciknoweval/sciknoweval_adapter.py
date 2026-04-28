from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# Only MCQ task types are wired here; the dataset also ships open-ended-qa,
# true_or_false, relation_extraction, and filling rows that need different
# scoring strategies (LLM judge or rule-based normalization). Those are
# skipped in record_to_sample by returning an empty list.
_MCQ_TYPES = {'mcq-4-choices', 'mcq-2-choices'}


@register_benchmark(
    BenchmarkMeta(
        name='sciknoweval',
        pretty_name='SciKnowEval',
        dataset_id='hicai-zju/SciKnowEval',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        few_shot_num=0,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class SciKnowEvalAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        if record.get('type') not in _MCQ_TYPES:
            return []
        choices = list(record['choices']['text'])
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answerKey'],
            metadata={
                'domain': record.get('domain'),
                'type': record.get('type'),
                'subtask': (record.get('details') or {}).get('subtask'),
            },
        )
