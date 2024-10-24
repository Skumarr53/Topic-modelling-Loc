from dataclasses import dataclass
from typing import Callable, Union, List
import ast

@dataclass
class Transformation:
    new_column: str
    columns_to_use: Union[str, List[str]]
    func: Callable



# Assuming 'topic', 'label', and 'label_column' are defined
trans_a = Transformation(
    f'{topic}_TOTAL_{label}',
    f'{label_column}_{label}',
    lambda x: x[topic]['total']
)

trans_b = Transformation(
    f'{topic}_STATS_{label}',
    f'{label_column}_{label}',
    lambda x: x[topic]['stats']
)

trans_c = Transformation(
    f'{topic}_REL_{label}',
    f'{topic}_TOTAL_{label}',
    lambda x: len([a for a in x if a > 0]) / len(x) if len(x) > 0 else None
)

trans_d = Transformation(
    f'{topic}_COUNT_{label}',
    f'{label_column}_{label}',
    lambda x: len([a for a in x if a > 0]) if len(x) > 0 else None
)

trans_e = Transformation(
    f'{topic}_EXTRACT_{label}',
    [label, f'{topic}_TOTAL_{label}'],
    lambda x: ' '.join([y for y, z in zip(x[0], x[1]) if z > 0])
)

trans_f = Transformation(
    f'{topic}_SENT_{label}',
    [f'{topic}_TOTAL_{label}', f'SENT_LABELS_{label}'],
    lambda x: calculate_sentence_score(x[0], x[1], weight=False)
)
