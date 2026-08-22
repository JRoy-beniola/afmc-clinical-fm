from collections.abc import Sequence

import numpy as np


def split_patient_ids(
    patient_ids: Sequence[str],
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("validation fraction must leave a non-empty test fraction")
    unique_ids = list(dict.fromkeys(patient_ids))
    if len(unique_ids) != len(patient_ids):
        raise ValueError("patient_ids must be unique")
    shuffled = np.asarray(unique_ids, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    val_end = train_end + int(len(shuffled) * val_fraction)
    return (
        shuffled[:train_end].tolist(),
        shuffled[train_end:val_end].tolist(),
        shuffled[val_end:].tolist(),
    )
