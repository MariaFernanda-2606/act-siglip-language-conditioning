from typing import Any

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_act_siglip import ACTSiglipConfig


def make_act_siglip_pre_post_processors(
    config: ACTSiglipConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Same normalisation pipeline as ACT.

    The language embedding is looked up inside the policy (from the batch `index` during training,
    or from `set_instruction` at inference), so no extra processing step is needed.
    """
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
