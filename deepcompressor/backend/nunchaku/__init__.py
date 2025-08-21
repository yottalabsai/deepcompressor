"""Nunchaku backend for DeepCompressor."""

# Enhanced metadata functionality - can be imported independently
from .convert_enhanced import (
    enhance_safetensors_with_config_metadata, 
    load_config_json, 
    prepare_metadata_from_config,
    resolve_actual_model_path,
    find_huggingface_cache_path,
    get_model_path_from_config,
    detect_precision_from_tensors,
    generate_quantization_config
)
from .validate_metadata import validate_metadata, load_safetensors_metadata

__all__ = [
    # Enhanced metadata functionality
    "enhance_safetensors_with_config_metadata",
    "load_config_json", 
    "prepare_metadata_from_config",
    "resolve_actual_model_path",
    "find_huggingface_cache_path", 
    "get_model_path_from_config",
    "detect_precision_from_tensors",
    "generate_quantization_config",
    "validate_metadata",
    "load_safetensors_metadata",
]
