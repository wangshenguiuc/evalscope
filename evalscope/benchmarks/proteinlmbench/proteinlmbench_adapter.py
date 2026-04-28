import re
import string
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_OPTION_NUM = re.compile(r'^\s*option\s*(\d+)', re.IGNORECASE)
_OPTION_PREFIX = re.compile(r'^\s*option\s*\d+\s*:\s*', re.IGNORECASE)


@register_benchmark(
    BenchmarkMeta(
        name='proteinlmbench',
        pretty_name='ProteinLMBench',
        dataset_id='tsynbio/ProteinLMBench',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        # Only the `evaluation` config is the MCQ benchmark; the other 7 configs
        # are large training corpora (UniProt instruction-tuning data, ~465k rows
        # per config) — out of scope for this rule-scored MCQ wire.
        subset_list=['evaluation'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ProteinLMBenchAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        options = record.get('options') or []
        if not options:
            return []
        choices = [_OPTION_PREFIX.sub('', o, count=1).strip() for o in options]
        m = _OPTION_NUM.match(record.get('answer') or '')
        if not m:
            return []
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(choices):
            return []
        target = string.ascii_uppercase[idx]
        return Sample(input=record['question'], choices=choices, target=target)
