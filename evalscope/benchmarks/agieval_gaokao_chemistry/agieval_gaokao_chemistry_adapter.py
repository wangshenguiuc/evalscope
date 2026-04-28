import re
import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_CHOICE_PREFIX = re.compile(r'^\s*\(?[A-Da-d]\)?\s*[\.、:：]?\s*')


def _strip_choice_prefix(text: str) -> str:
    return _CHOICE_PREFIX.sub('', text, count=1).strip()


@register_benchmark(
    BenchmarkMeta(
        name='agieval_gaokao_chemistry',
        pretty_name='AGIEval Gaokao Chemistry',
        dataset_id='hails/agieval-gaokao-chemistry',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AGIEvalGaokaoChemistryAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [_strip_choice_prefix(c) for c in record['choices']]
        gold = record['gold']
        idx = gold[0] if isinstance(gold, list) else int(gold)
        target = string.ascii_uppercase[int(idx)]
        query = record['query']
        m = re.search(r'问题[：:]\s*(.+?)\s*选项[：:]', query, flags=re.DOTALL)
        question = m.group(1).strip() if m else query
        return Sample(input=question, choices=choices, target=target)
