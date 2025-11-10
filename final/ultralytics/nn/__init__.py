# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# --- Custom Module Imports ---
# Add imports for the custom layers from the extra_modules directory.
# This makes them available to the model parser.
from .extra_modules.block import C2f_Faster_EMA, VoVGSCSP, GSConv
from .extra_modules.head import DetectAux
# C2PSA and SPPF are located in the standard nn/modules directory and should be loaded automatically.
# ---------------------------

from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    guess_model_scale,
    guess_model_task,
    load_checkpoint,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

__all__ = (
    "C2f_Faster_EMA",  # Expose custom module
    "VoVGSCSP",        # Expose custom module
    "GSConv",          # Expose custom module
    "DetectAux",       # Expose custom module
    "BaseModel",
    "ClassificationModel",
    "DetectionModel",
    "SegmentationModel",
    "guess_model_scale",
    "guess_model_task",
    "load_checkpoint",
    "parse_model",
    "torch_safe_load",
    "yaml_model_load",
)
