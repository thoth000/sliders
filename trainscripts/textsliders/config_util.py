from typing import Literal, Optional

import yaml

from pydantic import BaseModel
import torch

from lora import TRAINING_METHODS

PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
NETWORK_TYPES = Literal["lierla", "c3lier"]


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False

    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    type: NETWORK_TYPES = "lierla"
    rank: int = 4
    alpha: float = 1.0

    training_method: TRAINING_METHODS = "full"


class TrainConfig(BaseModel):
    precision: PRECISION_TYPES = "bfloat16"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a", "flow_match_euler_discrete"] = "ddim"

    iterations: int = 500
    lr: float = 1e-4
    optimizer: str = "adamw"
    optimizer_args: str = ""
    lr_scheduler: str = "constant"

    max_denoising_steps: int = 50
    
    # FLUX specific parameters
    height: Optional[int] = 512
    width: Optional[int] = 512
    max_sequence_length: Optional[int] = 512
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 3.5
    weighting_scheme: Optional[str] = "none"
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    mode_scale: Optional[float] = 1.29
    batch_size: Optional[int] = 1
    eta: Optional[float] = 2


class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 200
    precision: PRECISION_TYPES = "float32"


class LoggingConfig(BaseModel):
    use_wandb: bool = False

    verbose: bool = False


class OtherConfig(BaseModel):
    use_xformers: bool = False


class RootConfig(BaseModel):
    prompts_file: str
    pretrained_model: PretrainedModelConfig

    network: NetworkConfig

    train: Optional[TrainConfig]

    save: Optional[SaveConfig]

    logging: Optional[LoggingConfig]

    other: Optional[OtherConfig]


def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")


def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root
