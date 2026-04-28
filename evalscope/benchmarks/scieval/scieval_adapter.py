import re
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# Match the inline "\nA. …" answer-choices block at the end of a question.
_INLINE_CHOICES = re.compile(
    r'(?P<head>.*?)\n\n'
    r'A\.\s*(?P<a>.+?)\n'
    r'B\.\s*(?P<b>.+?)\n'
    r'C\.\s*(?P<c>.+?)\n'
    r'D\.\s*(?P<d>.+?)'
    r'(?:\n\nAnswer:.*)?$',
    re.DOTALL,
)


@register_benchmark(
    BenchmarkMeta(
        name='scieval',
        pretty_name='SciEval',
        dataset_id='OpenDFM/SciEval',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class SciEvalAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        # Filter to multiple-choice rows; other types (filling, judge) need
        # different scoring.
        if record.get('type') != 'multiple-choice':
            return []
        question = record.get('question') or ''
        m = _INLINE_CHOICES.match(question)
        if not m:
            return []
        stem = m.group('head').strip()
        choices = [m.group('a').strip(), m.group('b').strip(),
                   m.group('c').strip(), m.group('d').strip()]
        answer = record.get('answer')
        if isinstance(answer, list):
            if not answer:
                return []
            answer = answer[0]
        if not isinstance(answer, str) or len(answer) < 1 or answer[0].upper() not in 'ABCD':
            return []
        return Sample(
            input=stem,
            choices=choices,
            target=answer[0].upper(),
            metadata={
                'category': record.get('category'),
                'topic': record.get('topic'),
                'task_name': record.get('task_name'),
                'id': record.get('id'),
            },
        )
