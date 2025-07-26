"""
Nested dataclasses to store the Simulator configurations
"""

from dataclasses import dataclass


@dataclass
class Simulation:
    """
    Simulation-wide parameters
    """

    seed: int
    n_position: int
    n_variant_per_position: int


@dataclass
class ClassX:
    """
    Determines if a position is "null" or "mixed"

    Null: Models 1-2
    Mixed: Models 3-6
    """

    note: list[str]
    pi: list[float]
    class_type: list[str]


@dataclass
class MixedX:
    """
    For mixed positions, beta_x is sampled from a Gaussian mixture
    """

    pi: list[float]
    mu: list[float]
    omega: list[float]


@dataclass
class CausalGamma:
    """
    Causal effect
    """

    note: list[str]
    pi: list[float]
    mean: list[float]
    sd: list[float]


@dataclass
class CausalTau:
    """
    Position baseline effect
    """

    note: list[str]
    pi: list[float]
    mean: list[float]
    sd: list[float]


@dataclass
class Observation:
    """
    Uncertainty in the observation (compared with the true values)
    """

    sigma_x: float
    sigma_y: float
    sigma_theta: float


@dataclass
class Config:
    """
    Holds all configurations for the simulator
    """

    simulation: Simulation
    class_x: ClassX
    mixed_x: MixedX
    causal_gamma: CausalGamma
    causal_tau: CausalTau
    observation: Observation


DEFAULT_CONFIG = Config(
    simulation=Simulation(
        seed=1000,
        n_position=50,
        n_variant_per_position=20,
    ),
    class_x=ClassX(
        note=["model1_2", "model_3_4_5_6"],
        pi=[0.2, 0.8],
        class_type=["null", "mixed"],
    ),
    mixed_x=MixedX(
        pi=[0.4, 0.6],
        mu=[-2, 0],
        omega=[1, 0.5],
    ),
    causal_gamma=CausalGamma(
        note=["model3_4", "model5_6"],
        pi=[0.4, 0.6],
        mean=[0, 1],
        sd=[0, 0.3],
    ),
    causal_tau=CausalTau(
        note=["model1_3_5", "model2_4_6"],
        pi=[0.6, 0.4],
        mean=[0, -4],
        sd=[0, 2],
    ),
    observation=Observation(
        sigma_x=0.3,
        sigma_y=0.4,
        sigma_theta=0.05,
    ),
)


if __name__ == "__main__":

    print(DEFAULT_CONFIG)
