"""Utility script to validate metadata in safetensors files."""

import argparse
import json
import os
from typing import Dict, Any

try:
    import safetensors.torch
    from safetensors import safe_open
except ImportError:
    print("Warning: safetensors not installed. Validation functionality will not work.")
    safetensors = None
    safe_open = None


def load_safetensors_metadata(safetensors_path: str) -> Dict[str, str]:
    """Load metadata from a safetensors file.
    
    Args:
        safetensors_path (str): Path to the safetensors file
        
    Returns:
        Dict containing metadata
    """
    if safetensors is None or safe_open is None:
        print("Error: safetensors not available. Cannot load metadata.")
        return {}
        
    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
        return metadata or {}
    except Exception as e:
        print(f"Error loading metadata from {safetensors_path}: {e}")
        return {}


def validate_metadata(safetensors_path: str, verbose: bool = False) -> bool:
    """Validate that safetensors file contains expected metadata.
    
    Args:
        safetensors_path (str): Path to the safetensors file
        verbose (bool): Whether to print detailed information
        
    Returns:
        bool: True if metadata validation passes
    """
    if not os.path.exists(safetensors_path):
        print(f"Error: File not found: {safetensors_path}")
        return False
    
    print(f"Validating metadata in: {safetensors_path}")
    
    metadata = load_safetensors_metadata(safetensors_path)
    
    if not metadata:
        print("❌ No metadata found in the safetensors file")
        return False
    
    print(f"✅ Found metadata with {len(metadata)} fields")
    
    # Check for expected metadata fields
    expected_fields = [
        'metadata_source',
        'enhanced_by',
        'full_config'
    ]
    
    missing_fields = []
    for field in expected_fields:
        if field not in metadata:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"⚠️  Missing expected fields: {missing_fields}")
    else:
        print("✅ All expected metadata fields present")
    
    if verbose:
        print("\n=== Metadata Content ===")
        for key, value in metadata.items():
            if key == 'full_config':
                # Pretty print config JSON
                try:
                    config_data = json.loads(value)
                    print(f"{key}:")
                    print(json.dumps(config_data, indent=2)[:500] + "..." if len(value) > 500 else json.dumps(config_data, indent=2))
                except:
                    print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")
    
    # Validate full_config is valid JSON
    if 'full_config' in metadata:
        try:
            config_data = json.loads(metadata['full_config'])
            print("✅ full_config contains valid JSON")
            if verbose:
                print(f"Config contains {len(config_data)} top-level fields")
        except json.JSONDecodeError as e:
            print(f"❌ full_config contains invalid JSON: {e}")
            return False
    
    return True


def compare_metadata(file1_path: str, file2_path: str) -> None:
    """Compare metadata between two safetensors files.
    
    Args:
        file1_path (str): Path to first safetensors file
        file2_path (str): Path to second safetensors file
    """
    print(f"Comparing metadata between:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    
    metadata1 = load_safetensors_metadata(file1_path)
    metadata2 = load_safetensors_metadata(file2_path)
    
    if not metadata1 and not metadata2:
        print("Both files have no metadata")
        return
    elif not metadata1:
        print("File 1 has no metadata")
        return
    elif not metadata2:
        print("File 2 has no metadata")
        return
    
    # Compare keys
    keys1 = set(metadata1.keys())
    keys2 = set(metadata2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"\nCommon metadata fields: {len(common_keys)}")
    print(f"Only in file 1: {len(only_in_1)} - {list(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)} - {list(only_in_2)}")
    
    # Compare values for common keys
    different_values = []
    for key in common_keys:
        if metadata1[key] != metadata2[key]:
            different_values.append(key)
    
    if different_values:
        print(f"\nFields with different values: {different_values}")
    else:
        print("\n✅ All common fields have identical values")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Validate metadata in safetensors files")
    parser.add_argument("safetensors_path", type=str, 
                       help="Path to the safetensors file to validate")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed metadata information")
    parser.add_argument("--compare", type=str, default="",
                       help="Compare metadata with another safetensors file")
    
    args = parser.parse_args()
    
    # Validate primary file
    success = validate_metadata(args.safetensors_path, args.verbose)
    
    # Compare if requested
    if args.compare:
        print("\n" + "="*60)
        compare_metadata(args.safetensors_path, args.compare)
    
    if success:
        print("\n✅ Validation completed successfully!")
    else:
        print("\n❌ Validation failed!")
        exit(1)


if __name__ == "__main__":
    main()
