from dataclasses import dataclass

OBSERVATION_REGIMES = frozenset({"mcar", "mar", "mnar", "site_shift"})


@dataclass(frozen=True)
class SimulatorConfig:
    cohort_size: int = 1000
    n_sites: int = 2
    latent_dim: int = 4
    followup_days: float = 365.0
    mean_event_interval_days: float = 14.0
    process_noise: float = 0.03
    measurement_noise: float = 0.10
    intervention_rate: float = 0.04
    observation_regime: str = "mnar"
    world_name: str = "custom"
    heterogeneity_scale: float = 1.0
    delayed_intervention_effect: bool = False
    hidden_regime_switch: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "cohort_size": self.cohort_size,
            "n_sites": self.n_sites,
            "latent_dim": self.latent_dim,
            "followup_days": self.followup_days,
            "mean_event_interval_days": self.mean_event_interval_days,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.process_noise < 0 or self.measurement_noise < 0:
            raise ValueError("noise scales must be non-negative")
        if not 0 <= self.intervention_rate <= 1:
            raise ValueError("intervention_rate must be between 0 and 1")
        if self.heterogeneity_scale <= 0:
            raise ValueError("heterogeneity_scale must be positive")
        if self.observation_regime not in OBSERVATION_REGIMES:
            raise ValueError(f"unsupported observation_regime: {self.observation_regime}")
