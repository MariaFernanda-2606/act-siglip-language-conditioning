"""LeRobot policy plugin: ACT conditioned on SigLIP language embeddings.

Registers the policy type ``act_siglip`` so it can be used with
``lerobot-train --policy.type=act_siglip``.
"""

try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed; install lerobot to use this policy package.") from err

from .configuration_act_siglip import ACTSiglipConfig
from .modeling_act_siglip import ACTSiglipPolicy
from .planner import SubtaskPlanner
from .processor_act_siglip import make_act_siglip_pre_post_processors

__all__ = [
    "ACTSiglipConfig",
    "ACTSiglipPolicy",
    "SubtaskPlanner",
    "make_act_siglip_pre_post_processors",
]
