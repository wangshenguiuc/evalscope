import re
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# Earth-Silver MCQ rows ship the choices inline in the `question` field as
# `A) … B) … C) … D) …`. Choices are separated either by blank lines (`\n\n`)
# or by single spaces — match both with a non-greedy split. Anchored at "A)".
_INLINE = re.compile(
    r'\bA\)\s*(?P<a>.+?)\s+\bB\)\s*(?P<b>.+?)\s+'
    r'\bC\)\s*(?P<c>.+?)\s+\bD\)\s*(?P<d>.+?)$',
    re.DOTALL,
)


@register_benchmark(
    BenchmarkMeta(
        name='earth_silver',
        pretty_name='Earth-Silver',
        dataset_id='ai-earth/Earth-Silver',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        # Earth-Silver uses HF splits to subdivide task types; wire the
        # multiple_choice split. Free_form / fill_in_the_blank / true_false
        # would each need a different scoring strategy.
        few_shot_num=0,
        train_split=None,
        eval_split='multiple_choice',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class EarthSilverAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        question = record.get('question') or ''
        m = _INLINE.search(question)
        if not m:
            return []
        stem = question[:m.start()].rstrip().rstrip('.,;: ')
        if not stem:
            return []
        choices = [m.group(k).strip() for k in ('a', 'b', 'c', 'd')]
        answer = (record.get('answer') or '').strip()
        if not answer or answer[0].upper() not in 'ABCD':
            return []
        return Sample(
            input=stem,
            choices=choices,
            target=answer[0].upper(),
            metadata={
                'task': record.get('task'),
                'sphere': record.get('sphere'),
                'subject': record.get('subject'),
                'sub_discipline': record.get('sub_discipline'),
                'idx': record.get('idx'),
            },
        )
