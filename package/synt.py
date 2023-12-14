from collections import Counter
from typing import List, Dict


def pos_rates(pos: List[str]) -> Dict[str, float]:
    return {k: v / len(pos) for k, v in Counter(pos).items()}