#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced Nunchaku convert functionality
with config.json metadata support.

This script shows three different ways to add metadata to safetensors files:
1. Using the enhanced convert.py script directly
2. Using the convert_enhanced module functions
3. Post-processing existing safetensors files

Author: DeepCompressor Team
"""

import os
import sys
from pathlib import Path

# Add deepcompressor to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepcompressor.backend.nunchaku.convert_enhanced import (
    enhance_safetensors_with_config_metadata,
    load_config_json,
    prepare_metadata_from_config
)
from deepcompressor.backend.nunchaku.validate_metadata import validate_metadata


def example_1_enhanced_convert_script():
    """Example 1: Using the enhanced convert.py script with metadata support."""
    print("=== Example 1: Enhanced Convert Script ===")
    print("To use the enhanced convert script with metadata support:")
    print()
    
    example_cmd = """
python -m deepcompressor.backend.nunchaku.convert \\
    --quant-path /path/to/quantization/checkpoint \\
    --output-root /path/to/output \\
    --model-name flux.1-dev \\
    --add-metadata \\
    --config-yaml examples/diffusion/configs/model/flux.1-dev.yaml \\
    --model-path /path/to/flux.1-dev/model
"""
    
    print("Command:")
    print(example_cmd)
    print()
    print("Key parameters:")
    print("  --add-metadata         : Enable metadata extraction and addition")
    print("  --config-yaml         : Path to diffusion config YAML file")
    print("  --model-path          : Path to model directory (optional, overrides config detection)")
    print()


def example_2_programmatic_usage():
    """Example 2: Using the metadata functions programmatically."""
    print("=== Example 2: Programmatic Usage ===")
    print("To use the metadata functions in your own code:")
    print()
    
    code_example = '''
from deepcompressor.backend.nunchaku.convert_enhanced import (
    load_config_json, prepare_metadata_from_config
)
import safetensors.torch

# Load config.json from model directory
model_path = "/path/to/flux.1-dev/model"
config_data = load_config_json(model_path)

if config_data:
    # Prepare metadata for safetensors
    metadata = prepare_metadata_from_config(config_data)
    
    # Your tensor data
    tensors_dict = {
        "layer1.weight": your_tensor_data,
        # ... more tensors
    }
    
    # Save with metadata
    safetensors.torch.save_file(
        tensors_dict, 
        "output.safetensors", 
        metadata=metadata
    )
    print("Saved safetensors file with metadata!")
'''
    
    print("Code example:")
    print(code_example)
    print()


def example_3_post_processing():
    """Example 3: Post-processing existing safetensors files."""
    print("=== Example 3: Post-Processing Existing Files ===")
    print("To add metadata to existing safetensors files:")
    print()
    
    code_example = '''
from deepcompressor.backend.nunchaku.convert_enhanced import (
    enhance_safetensors_with_config_metadata
)

# Enhance existing safetensors file
success = enhance_safetensors_with_config_metadata(
    safetensors_path="/path/to/transformer_blocks.safetensors",
    config_yaml_path="examples/diffusion/configs/model/flux.1-dev.yaml",
    model_path_override="/path/to/flux.1-dev/model"  # optional
)

if success:
    print("Successfully enhanced safetensors file with metadata!")
else:
    print("Enhancement failed!")
'''
    
    print("Code example:")
    print(code_example)
    print()
    
    print("Command line usage:")
    cmd_example = """
python -m deepcompressor.backend.nunchaku.convert_enhanced \\
    --safetensors-path /path/to/transformer_blocks.safetensors \\
    --config-yaml examples/diffusion/configs/model/flux.1-dev.yaml \\
    --model-path /path/to/flux.1-dev/model
"""
    print(cmd_example)
    print()


def example_4_validation():
    """Example 4: Validating metadata in safetensors files."""
    print("=== Example 4: Validating Metadata ===")
    print("To validate that metadata was correctly added:")
    print()
    
    cmd_example = """
# Basic validation
python -m deepcompressor.backend.nunchaku.validate_metadata \\
    /path/to/transformer_blocks.safetensors

# Verbose validation (shows metadata content)
python -m deepcompressor.backend.nunchaku.validate_metadata \\
    /path/to/transformer_blocks.safetensors --verbose

# Compare metadata between two files
python -m deepcompressor.backend.nunchaku.validate_metadata \\
    /path/to/file1.safetensors --compare /path/to/file2.safetensors
"""
    
    print("Command examples:")
    print(cmd_example)
    print()
    
    code_example = '''
from deepcompressor.backend.nunchaku.validate_metadata import (
    validate_metadata, load_safetensors_metadata
)

# Validate metadata
is_valid = validate_metadata("/path/to/transformer_blocks.safetensors", verbose=True)

# Load and inspect metadata manually
metadata = load_safetensors_metadata("/path/to/transformer_blocks.safetensors")
print(f"Found {len(metadata)} metadata fields:")
for key, value in metadata.items():
    print(f"  {key}: {value[:100]}...")
'''
    
    print("Programmatic validation:")
    print(code_example)
    print()


def example_5_typical_workflow():
    """Example 5: Typical workflow for using the enhanced functionality."""
    print("=== Example 5: Typical Workflow ===")
    print("Complete workflow for quantization with metadata:")
    print()
    
    workflow = """
1. Prepare your diffusion model config (flux.1-dev.yaml)
2. Run quantization to generate model checkpoints
3. Convert with enhanced metadata support:

   python -m deepcompressor.backend.nunchaku.convert \\
       --quant-path ./quantization_output \\
       --output-root ./nunchaku_output \\
       --model-name flux.1-dev \\
       --add-metadata \\
       --config-yaml examples/diffusion/configs/model/flux.1-dev.yaml \\
       --model-path /path/to/huggingface/flux.1-dev

4. Validate the enhanced files:

   python -m deepcompressor.backend.nunchaku.validate_metadata \\
       ./nunchaku_output/flux.1-dev/transformer_blocks.safetensors --verbose

5. Use the enhanced safetensors files in your application
"""
    
    print(workflow)
    print()


def show_metadata_structure():
    """Show the structure of metadata that gets added."""
    print("=== Metadata Structure ===")
    print("The metadata added to safetensors files includes:")
    print()
    
    structure = """
{
  "model_type": "flux",
  "architectures": ["FluxTransformer2DModel"],
  "hidden_size": "3072",
  "num_hidden_layers": "19",
  "num_attention_heads": "24", 
  "intermediate_size": "12288",
  "max_position_embeddings": "512",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.0",
  "full_config": "{...complete config.json as JSON string...}",
  "metadata_source": "config.json",
  "enhanced_by": "deepcompressor_nunchaku_convert_enhanced"
}
"""
    
    print("Example metadata structure:")
    print(structure)
    print()
    print("Notes:")
    print("- All values are stored as strings (safetensors requirement)")
    print("- 'full_config' contains the complete original config.json")
    print("- Additional fields from config.json are included automatically")
    print()


def main():
    """Main function to display all examples."""
    print("=" * 80)
    print("Enhanced Nunchaku Convert with Config.json Metadata Support")
    print("=" * 80)
    print()
    
    example_1_enhanced_convert_script()
    print("\n" + "-" * 60 + "\n")
    
    example_2_programmatic_usage()
    print("\n" + "-" * 60 + "\n")
    
    example_3_post_processing()
    print("\n" + "-" * 60 + "\n")
    
    example_4_validation()
    print("\n" + "-" * 60 + "\n")
    
    example_5_typical_workflow()
    print("\n" + "-" * 60 + "\n")
    
    show_metadata_structure()
    print("\n" + "=" * 80)
    print("For more information, see the module documentation:")
    print("- deepcompressor.backend.nunchaku.convert_enhanced")
    print("- deepcompressor.backend.nunchaku.validate_metadata")
    print("=" * 80)


if __name__ == "__main__":
    main()
