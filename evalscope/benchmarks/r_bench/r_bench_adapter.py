from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

_LETTERS = ('A', 'B', 'C', 'D', 'E', 'F')


@register_benchmark(
    BenchmarkMeta(
        name='r_bench',
        pretty_name='R-Bench',
        dataset_id='R-Bench/R-Bench',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        # Text-only subsets; rbench-m_* (multimodal) excluded — out of scope for qwen3.5-35b-a3b smoke
        subset_list=['rbench-t_en', 'rbench-t_zh'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class RBenchAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record[k] for k in _LETTERS]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={'index': record.get('index')},
        )
