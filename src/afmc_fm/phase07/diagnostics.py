from __future__ import annotations

import pandas as pd


def first_patience_exhaustion_epoch(
    trace: pd.DataFrame,
    *,
    patience: int,
) -> int | None:
    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")
    if type(patience) is not int or patience <= 0:
        raise ValueError("patience must be a positive integer")
    missing = {"epoch", "stale_epochs"} - set(trace.columns)
    if missing:
        raise ValueError(
            "trace is missing columns: " + ", ".join(sorted(missing))
        )
    if trace.empty:
        return None

    epochs = pd.to_numeric(trace["epoch"], errors="raise")
    stale = pd.to_numeric(trace["stale_epochs"], errors="raise")
    matched = trace.loc[stale >= patience]
    if matched.empty:
        return None
    return int(pd.to_numeric(matched.iloc[0]["epoch"], errors="raise"))


__all__ = ["first_patience_exhaustion_epoch"]
