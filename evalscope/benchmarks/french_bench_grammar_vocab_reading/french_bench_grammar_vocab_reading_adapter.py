from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='french_bench_grammar_vocab_reading',
        pretty_name='French-Bench (Grammar / Vocab / Reading)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

French-Bench Grammar/Vocab/Reading (`manu/french-bench-grammar-vocab-reading`)
is a 309-question French-language MCQ benchmark spread across three
HF *splits* (not configs): Grammar (119), Vocabulary (119), Reading
(71).

## Task Description

- **Task Type**: 4-choice French-language MCQ
- **Input**: `question` plus literal `answerA`/`B`/`C`/`D` columns
- **Output**: Single correct answer letter (A/B/C/D)

## Evaluation Notes

- The dataset uses a single `default` config but distinct splits per
  subject (`Grammar`, `Vocabulary`, `Reading`). evalscope's
  `eval_split` is single-valued, so this row evaluates the
  `Grammar` split — the most representative French knowledge probe
  and the largest split. Vocabulary and Reading require separate
  wires if the human reviewer wants full coverage.
- Primary metric: **Accuracy**.
""",
        dataset_id='manu/french-bench-grammar-vocab-reading',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='Grammar',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class FrenchBenchGrammarVocabReadingAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['answerA'], record['answerB'],
            record['answerC'], record['answerD'],
        ]
        question = record['question']
        context = record.get('context')
        if context and context != 'No Context':
            question = f'{context}\n\n{question}'
        return Sample(
            input=question,
            choices=choices,
            target=record['answer'],
            metadata={
                'subject': record.get('subject'),
                'difficulty': record.get('difficulty'),
            },
        )
