import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_INLINE_CHOICES = re.compile(r'\n[A-D]\.\s')


@register_benchmark(
    BenchmarkMeta(
        name='medconceptsqa',
        pretty_name='MedConceptsQA',
        dataset_id='ofir408/MedConceptsQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        # Use the aggregate `all` config (15 vocab×level slices fully merged); 819k test rows
        # are capped to a tractable smoke via fast_limit at the BENCHMARKS layer.
        subset_list=['all'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MedConceptsQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        # Strip the inline "\nA. … \nB. …" answer-choices block; choices are
        # passed separately to let the parent template render them.
        m = _INLINE_CHOICES.search(question)
        if m:
            question = question[:m.start()].rstrip()
        choices = [record['option1'], record['option2'],
                   record['option3'], record['option4']]
        return Sample(
            input=question,
            choices=choices,
            target=record['answer_id'],
            metadata={
                'vocab': record.get('vocab'),
                'level': record.get('level'),
                'question_id': record.get('question_id'),
            },
        )
