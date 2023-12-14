import numpy as np
from typing import List, Optional


def TTR(words: List[str]) -> float:
    return len(set(words)) / len(words)


def MATTR(words: List[str], w: Optional[int] = 10) -> float:
    if len(words) <= w:
        return TTR(words)
    else:
        ttrs = []
        for i in range(len(words) - w):
            ttrs.append(TTR(words[i:i + w]))
        return np.mean(ttrs)
