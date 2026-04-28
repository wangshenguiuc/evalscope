import json
import string
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='chembench',
        pretty_name='ChemBench',
        dataset_id='jablonkagroup/ChemBench',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        subset_list=[
            'analytical_chemistry', 'chemical_preference', 'general_chemistry',
            'inorganic_chemistry', 'materials_science', 'organic_chemistry',
            'physical_chemistry', 'technical_chemistry', 'toxicity_and_safety',
        ],
        few_shot_num=0,
        train_split='train',
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class ChemBenchAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        # Skip non-MCQ rows (numeric / exact-match metrics).
        if 'multiple_choice_grade' not in (record.get('metrics') or []):
            return []
        examples = record.get('examples') or []
        if not examples:
            return []
        ex = examples[0]
        question = ex.get('input')
        scores_json = ex.get('target_scores')
        if not question or not scores_json:
            return []
        try:
            scores = json.loads(scores_json)
        except (TypeError, ValueError):
            return []
        if not isinstance(scores, dict) or not scores:
            return []
        choice_texts = list(scores.keys())
        # Accept both 1 and 1.0 as the correct marker.
        correct = [t for t, s in scores.items() if float(s) >= 1.0]
        if not correct:
            return []
        target = string.ascii_uppercase[choice_texts.index(correct[0])]
        return Sample(
            input=question,
            choices=choice_texts,
            target=target,
            metadata={
                'subfield': record.get('subfield'),
                'name': record.get('name'),
                'uuid': record.get('uuid'),
            },
        )
