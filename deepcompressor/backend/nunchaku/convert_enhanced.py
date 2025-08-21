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
        
        # If no path specified, use model name as default
        if not model_path:
            model_name = pipeline_config.get('name', '')
            if model_name:
                # For models like flux.1-dev, try common paths
                print(f"No explicit path found, using model name: {model_name}")
                return model_name
        
        return model_path if model_path else None
        
    except Exception as e:
        print(f"Error loading config yaml from {config_yaml_path}: {e}")
        return None


def prepare_metadata_from_config(config_data: Dict[str, Any]) -> Dict[str, str]:
    """Prepare metadata dictionary for safetensors from config.json data.
    
    Args:
        config_data (Dict): Data loaded from config.json
        
    Returns:
        Dict with string keys and string values for safetensors metadata
    """
    # safetensors metadata must have string values
    metadata = {}
    
    # Common fields to include in metadata
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
    
    # Add full config as JSON string for complete preservation
    metadata['full_config'] = json.dumps(config_data, separators=(',', ':'))
    
    # Add metadata source info
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
    
    # Load config.json
    config_data = load_config_json(model_path)
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
    
    # Prepare metadata
    metadata = prepare_metadata_from_config(config_data)
    
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
