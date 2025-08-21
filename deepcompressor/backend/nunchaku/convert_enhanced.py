"""Enhances convert functionality to add metadata from config.json to safetensors files."""

import json
import os
import argparse
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. YAML config parsing will not work.")
    yaml = None

try:
    import safetensors.torch
except ImportError:
    print("Warning: safetensors not installed. Core functionality will not work.")
    safetensors = None


def load_config_json(model_path: str) -> Optional[Dict[str, Any]]:
    """Load config.json from model transformer directory.
    
    Args:
        model_path (str): Path to the model directory
        
    Returns:
        Dict containing config.json data or None if not found
    """
    config_path = os.path.join(model_path, "transformer", "config.json")
    if not os.path.exists(config_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(model_path, "config.json"),
            os.path.join(model_path, "transformer.config.json"),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
        else:
            print(f"Warning: config.json not found in {model_path}/transformer/ or alternative paths")
            return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"Successfully loaded config.json from {config_path}")
        return config_data
    except Exception as e:
        print(f"Error loading config.json from {config_path}: {e}")
        return None


def get_model_path_from_config(config_yaml_path: str) -> Optional[str]:
    """Extract model path from diffusion config yaml file.
    
    Args:
        config_yaml_path (str): Path to the config yaml file
        
    Returns:
        Model path or None if not found
    """
    if yaml is None:
        print("Error: PyYAML not available. Cannot parse YAML config file.")
        return None
        
    try:
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check if path is specified in pipeline config
        pipeline_config = config.get('pipeline', {})
        model_path = pipeline_config.get('path', '')
        model_name = pipeline_config.get('name', '')
        
        # If explicit path is provided, use it
        if model_path:
            print(f"Found explicit path in config: {model_path}")
            return model_path
        
        # If no path specified, map model name to default HuggingFace path
        if model_name:
            default_paths = {
                "flux.1-dev": "black-forest-labs/FLUX.1-dev",
                "flux.1-canny-dev": "black-forest-labs/FLUX.1-Canny-dev", 
                "flux.1-depth-dev": "black-forest-labs/FLUX.1-Depth-dev",
                "flux.1-fill-dev": "black-forest-labs/FLUX.1-Fill-dev",
                "flux.1-schnell": "black-forest-labs/FLUX.1-schnell",
                "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
                "sdxl-turbo": "stabilityai/sdxl-turbo",
                "pixart-sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            }
            
            default_path = default_paths.get(model_name)
            if default_path:
                print(f"Mapped model name '{model_name}' to default path: {default_path}")
                return default_path
            else:
                print(f"No default path mapping found for model: {model_name}")
                return model_name
        
        return None
        
    except Exception as e:
        print(f"Error loading config yaml from {config_yaml_path}: {e}")
        return None


def find_huggingface_cache_path(model_id: str) -> Optional[str]:
    """Find the actual cache path for a HuggingFace model.
    
    Args:
        model_id (str): HuggingFace model ID (e.g., "black-forest-labs/FLUX.1-dev")
        
    Returns:
        Actual local cache path or None if not found
    """
    import os
    from pathlib import Path
    
    # Try to get HuggingFace cache directory
    try:
        from huggingface_hub import cached_download, hf_hub_download
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        cache_dir = HUGGINGFACE_HUB_CACHE
    except ImportError:
        # Fallback to default cache location
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    print(f"Searching for model '{model_id}' in HuggingFace cache: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print(f"HuggingFace cache directory not found: {cache_dir}")
        return None
    
    # HuggingFace cache structure: models--{org}--{model_name}
    # Convert model_id to cache directory name
    cache_model_name = f"models--{model_id.replace('/', '--')}"
    model_cache_dir = os.path.join(cache_dir, cache_model_name)
    
    print(f"Looking for cached model at: {model_cache_dir}")
    
    if not os.path.exists(model_cache_dir):
        print(f"Model cache directory not found: {model_cache_dir}")
        return None
    
    # Look for snapshots directory and get the latest snapshot
    snapshots_dir = os.path.join(model_cache_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        print(f"Snapshots directory not found: {snapshots_dir}")
        return None
    
    # Find the most recent snapshot (usually the only one)
    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshots:
        print(f"No snapshots found in: {snapshots_dir}")
        return None
    
    # Use the first snapshot (there's usually only one)
    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
    print(f"Found model snapshot at: {snapshot_path}")
    
    # Verify that config.json or transformer/config.json exists
    config_paths = [
        os.path.join(snapshot_path, "transformer", "config.json"),
        os.path.join(snapshot_path, "config.json"),
        os.path.join(snapshot_path, "transformer.config.json"),
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"Found config.json at: {config_path}")
            return snapshot_path
    
    print(f"No config.json found in snapshot: {snapshot_path}")
    return None


def resolve_actual_model_path(model_path: str) -> Optional[str]:
    """Resolve the actual local path for a model, whether it's a local path or HuggingFace ID.
    
    Args:
        model_path (str): Model path (local path or HuggingFace model ID)
        
    Returns:
        Actual local path where config.json can be found
    """
    # If it's already a local path that exists, use it directly
    if os.path.isdir(model_path):
        print(f"Using local path: {model_path}")
        return model_path
    
    # If it looks like a HuggingFace model ID (contains '/'), try to find it in cache
    if '/' in model_path:
        cache_path = find_huggingface_cache_path(model_path)
        if cache_path:
            return cache_path
    
    # Try to treat as local path anyway
    if os.path.isdir(model_path):
        return model_path
    
    print(f"Could not resolve model path: {model_path}")
    return None


def detect_precision_from_tensors(state_dict: Dict[str, Any]) -> str:
    """Detect precision (int4/fp4) from tensor dtypes.
    
    Args:
        state_dict (Dict): Dictionary of tensors
        
    Returns:
        str: "fp4" if float8 dtypes found, "int4" otherwise
    """
    try:
        import torch
        fp8_dtypes = [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz, 
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e8m0fnu,
        ]
        
        for tensor in state_dict.values():
            if hasattr(tensor, 'dtype') and tensor.dtype in fp8_dtypes:
                return "fp4"
        return "int4"
    except (ImportError, AttributeError):
        # Fallback to int4 if torch not available or dtypes not supported
        return "int4"


def generate_quantization_config(precision: str = "int4") -> Dict[str, Any]:
    """Generate quantization config for nunchaku.
    
    Args:
        precision (str): "int4" or "fp4"
        
    Returns:
        Dict containing quantization configuration
    """
    return {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4", 
            "scale_dtype": "fp8_e4m3_nan" if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
    }


def prepare_metadata_from_config(config_data: Dict[str, Any], 
                                state_dict: Dict[str, Any] = None,
                                model_class: str = "NunchakuFluxTransformer2dModel") -> Dict[str, str]:
    """Prepare metadata dictionary for safetensors from config.json data.
    
    Args:
        config_data (Dict): Data loaded from config.json
        state_dict (Dict, optional): Model state dict for precision detection
        model_class (str): Model class name for nunchaku
        
    Returns:
        Dict with string keys and string values for safetensors metadata
    """
    # safetensors metadata must have string values
    metadata = {}
    
    # Common fields to include in metadata (for backward compatibility)
    important_fields = [
        'model_type', 'architectures', 'hidden_size', 'num_hidden_layers',
        'num_attention_heads', 'intermediate_size', 'max_position_embeddings',
        'torch_dtype', 'transformers_version', 'num_key_value_heads',
        'rope_theta', 'attention_dropout', 'hidden_act'
    ]
    
    for field in important_fields:
        if field in config_data:
            value = config_data[field]
            # Convert to string, handling different types
            if isinstance(value, (list, dict)):
                metadata[field] = json.dumps(value)
            else:
                metadata[field] = str(value)
    
    # Prepare full config JSON
    full_config_json = json.dumps(config_data, separators=(',', ':'))
    
    # CRITICAL: Add the 'config' field that nunchaku requires
    metadata['config'] = full_config_json
    
    # Add model class for nunchaku
    metadata['model_class'] = model_class
    
    # Generate and add quantization config
    precision = "int4"  # default
    if state_dict:
        precision = detect_precision_from_tensors(state_dict)
        print(f"Detected precision: {precision}")
    
    quantization_config = generate_quantization_config(precision)
    metadata['quantization_config'] = json.dumps(quantization_config, separators=(',', ':'))
    
    # Add optional comfy_config (empty for now, can be extended later)
    metadata['comfy_config'] = json.dumps({}, separators=(',', ':'))
    
    # Legacy fields for compatibility
    metadata['full_config'] = full_config_json
    metadata['metadata_source'] = 'config.json'
    metadata['enhanced_by'] = 'deepcompressor_nunchaku_convert_enhanced'
    
    return metadata


def save_safetensors_with_metadata(tensors_dict: Dict[str, Any], 
                                   output_path: str, 
                                   metadata: Dict[str, str]) -> None:
    """Save tensors to safetensors file with metadata.
    
    Args:
        tensors_dict (Dict): Dictionary of tensors to save
        output_path (str): Path to save the safetensors file
        metadata (Dict): Metadata dictionary with string values
    """
    if safetensors is None:
        raise ImportError("safetensors not available. Cannot save safetensors file.")
    
    print(f"Saving safetensors file to {output_path} with metadata...")
    print(f"Metadata keys: {list(metadata.keys())}")
    
    safetensors.torch.save_file(tensors_dict, output_path, metadata=metadata)
    print(f"Successfully saved safetensors file with metadata to {output_path}")


def enhance_safetensors_with_config_metadata(safetensors_path: str, 
                                            config_yaml_path: str,
                                            model_path_override: Optional[str] = None) -> bool:
    """Enhance existing safetensors file by adding metadata from config.json.
    
    Args:
        safetensors_path (str): Path to the safetensors file to enhance
        config_yaml_path (str): Path to the diffusion config yaml file
        model_path_override (str, optional): Override model path if provided
        
    Returns:
        bool: True if enhancement was successful, False otherwise
    """
    if not os.path.exists(safetensors_path):
        print(f"Error: Safetensors file not found: {safetensors_path}")
        return False
    
    # Get model path
    if model_path_override:
        model_path = model_path_override
    else:
        model_path = get_model_path_from_config(config_yaml_path)
        if not model_path:
            print(f"Error: Could not determine model path from {config_yaml_path}")
            return False
    
    print(f"Using model path: {model_path}")
    
    # Resolve the actual local path (handle HuggingFace model IDs)
    actual_model_path = resolve_actual_model_path(model_path)
    if not actual_model_path:
        print(f"Error: Could not resolve actual model path for: {model_path}")
        return False
    
    print(f"Resolved to actual path: {actual_model_path}")
    
    # Load config.json
    config_data = load_config_json(actual_model_path)
    if not config_data:
        print("Error: Could not load config.json")
        return False
    
    # Load existing safetensors file
    try:
        if safetensors is None:
            print("Error: safetensors not available. Cannot load safetensors file.")
            return False
            
        print(f"Loading existing safetensors file: {safetensors_path}")
        tensors_dict = safetensors.torch.load_file(safetensors_path)
        print(f"Loaded {len(tensors_dict)} tensors")
    except Exception as e:
        print(f"Error loading safetensors file {safetensors_path}: {e}")
        return False
    
    # Prepare metadata (pass tensors_dict for precision detection)
    metadata = prepare_metadata_from_config(config_data, tensors_dict)
    
    # Create backup
    backup_path = safetensors_path + ".backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(safetensors_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Save enhanced file
    try:
        save_safetensors_with_metadata(tensors_dict, safetensors_path, metadata)
        return True
    except Exception as e:
        print(f"Error saving enhanced safetensors file: {e}")
        return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Enhance safetensors files with config.json metadata")
    parser.add_argument("--safetensors-path", type=str, required=True,
                       help="Path to the safetensors file to enhance")
    parser.add_argument("--config-yaml", type=str, required=True,
                       help="Path to the diffusion config yaml file")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model path (optional)")
    
    args = parser.parse_args()
    
    success = enhance_safetensors_with_config_metadata(
        args.safetensors_path,
        args.config_yaml,
        args.model_path
    )
    
    if success:
        print("Enhancement completed successfully!")
    else:
        print("Enhancement failed!")
        exit(1)


if __name__ == "__main__":
    main()
