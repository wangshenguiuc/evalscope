import string
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# 62 subsets shipped by HiTZ/EusExams: Basque (eu_*) and Spanish (es_*)
# professional/civil-service exam questions across health-system,
# university administration, justice, and municipal job categories.
SUBSET_LIST = [
    'eu_opeosakiadmineu', 'eu_opeosakiauxenfeu', 'eu_opeosakiauxeu',
    'eu_opeosakiceladoreu', 'eu_opeosakienfeu',
    'eu_opeosakioperarioeu', 'eu_opeosakitecnicoeu',
    'eu_opeosakivarioseu', 'eu_opegasteizkoudala',
    'eu_opeehuadmineu', 'eu_opeehuauxeu', 'eu_opeehubiblioeu',
    'eu_opeehuderechoeu', 'eu_opeehueconomicaseu',
    'eu_opeehuempresarialeseu', 'eu_opeehusubalternoeu',
    'eu_opeehutecnicoeu', 'eu_opeehuteknikarib', 'eu_ejadministrari',
    'eu_ejlaguntza', 'eu_ejlaguntzaile', 'eu_ejteknikari',
    'eu_osakidetza1e', 'eu_osakidetza2e', 'eu_osakidetza3e',
    'eu_osakidetza5e', 'eu_osakidetza6e', 'eu_osakidetza7e',
    'eu_opebilbaoeu',
    'es_opeosakiadmin', 'es_opeosakiaux', 'es_opeosakiauxenf',
    'es_opeosakicelador', 'es_opeosakienf', 'es_opeosakijuridico',
    'es_opeosakioperario', 'es_opeosakitecnico',
    'es_opeosakivarios', 'es_opeayuntamientovitoria',
    'es_opeehuadmin', 'es_opeehuaux', 'es_opeehubiblio',
    'es_opeehuderecho', 'es_opeehueconomicas',
    'es_opeehuempresariales', 'es_opeehusubalterno',
    'es_opeehutecnico', 'es_opeehutecnicob',
    'es_ejadministrativo', 'es_ejauxiliar', 'es_ejsubalterno',
    'es_ejtecnico', 'es_osakidetza1c', 'es_osakidetza2c',
    'es_osakidetza3c', 'es_osakidetza4c', 'es_osakidetza5c',
    'es_osakidetza6c', 'es_osakidetza7c', 'es_osakidetza8c',
    'es_osakidetza9c', 'es_opebilbao',
]


@register_benchmark(
    BenchmarkMeta(
        name='eusexams',
        pretty_name='EusExams',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

EusExams (`HiTZ/EusExams`) is a Basque / Spanish professional and
civil-service exam MCQ benchmark spanning 62 subsets — eu_* and es_*
mirror translations of the same exam questions (health-system,
university administration, justice, municipal job categories).

## Task Description

- **Task Type**: Basque / Spanish 4-choice MCQ
- **Input**: `question` plus `candidates` (list of choice strings)
- **Output**: Single correct answer letter (A/B/C/D); the dataset
  stores `answer` as a 0-indexed int.

## Evaluation Notes

- 62 per-exam configs; per-subset sizes typically 100–400 rows.
- Primary metric: **Accuracy**.
""",
        dataset_id='HiTZ/EusExams',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class EusExamsAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = list(record['candidates'])
        ans = record.get('answer')
        if ans is None:
            target = ''
        else:
            target = string.ascii_uppercase[int(ans)]
        return Sample(
            input=record['question'],
            choices=choices,
            target=target,
            metadata={'id': record.get('id')},
        )
