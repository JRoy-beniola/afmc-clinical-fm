import pandas as pd

from afmc_fm.models.baselines import GRUBaseline, TorchMLPRegressorBaseline

_MIN_HIDDEN_SIZE = 1
_MAX_HIDDEN_SIZE = 256


def _validate_positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _gru_parameter_count(*, value_dim: int, event_dim: int, hidden_size: int) -> int:
    model = GRUBaseline(
        value_dim=value_dim,
        event_dim=event_dim,
        hidden_size=hidden_size,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def _mlp_parameter_count(*, input_dim: int, hidden_size: int) -> int:
    return TorchMLPRegressorBaseline(
        input_dim=input_dim,
        seed=0,
        device="cpu",
        hidden_size=hidden_size,
    ).trainable_parameter_count()


def _closest_hidden_size(target_parameters: int, counter) -> int:
    return min(
        range(_MIN_HIDDEN_SIZE, _MAX_HIDDEN_SIZE + 1),
        key=lambda width: (abs(counter(width) - target_parameters), width),
    )


def closest_gru_hidden_size(
    target_parameters: int,
    value_dim: int,
    event_dim: int,
) -> int:
    _validate_positive(
        target_parameters=target_parameters,
        value_dim=value_dim,
        event_dim=event_dim,
    )
    return _closest_hidden_size(
        target_parameters,
        lambda width: _gru_parameter_count(
            value_dim=value_dim,
            event_dim=event_dim,
            hidden_size=width,
        ),
    )


def closest_mlp_hidden_size(target_parameters: int, input_dim: int) -> int:
    _validate_positive(target_parameters=target_parameters, input_dim=input_dim)
    return _closest_hidden_size(
        target_parameters,
        lambda width: _mlp_parameter_count(
            input_dim=input_dim,
            hidden_size=width,
        ),
    )


def build_capacity_audit(
    *,
    target_parameters: int,
    value_dim: int,
    event_dim: int,
    representation_input_dim: int,
) -> pd.DataFrame:
    _validate_positive(
        target_parameters=target_parameters,
        value_dim=value_dim,
        event_dim=event_dim,
        representation_input_dim=representation_input_dim,
    )

    gru_hidden = closest_gru_hidden_size(
        target_parameters,
        value_dim=value_dim,
        event_dim=event_dim,
    )
    mlp_hidden = closest_mlp_hidden_size(
        target_parameters,
        input_dim=representation_input_dim,
    )
    rows = []
    for control, hidden_size, actual_parameters in (
        (
            "matched_gru",
            gru_hidden,
            _gru_parameter_count(
                value_dim=value_dim,
                event_dim=event_dim,
                hidden_size=gru_hidden,
            ),
        ),
        (
            "matched_representation_mlp",
            mlp_hidden,
            _mlp_parameter_count(
                input_dim=representation_input_dim,
                hidden_size=mlp_hidden,
            ),
        ),
    ):
        absolute_mismatch = abs(actual_parameters - target_parameters)
        rows.append(
            {
                "control": control,
                "target_parameters": target_parameters,
                "hidden_size": hidden_size,
                "actual_parameters": actual_parameters,
                "absolute_mismatch": absolute_mismatch,
                "relative_mismatch": absolute_mismatch / target_parameters,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "build_capacity_audit",
    "closest_gru_hidden_size",
    "closest_mlp_hidden_size",
]
