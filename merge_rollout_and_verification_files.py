import json
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def check_for_collisions(verification_solutions: List[Dict], 
                         full_raw_rollout_data_array: List[Dict]) -> Tuple[List, List, bool, bool]:
    """
    Check for collision errors between verification solutions and rollout data.
    
    Args:
        verification_solutions: List of verification solution dicts
        full_raw_rollout_data_array: List of rollout data dicts
        
    Returns:
        Tuple of (collision_errors, no_matches_array, has_collisions, has_no_matches)
    """
    # Create a mapping of unique_key to count for verification solutions
    verification_key_counts = defaultdict(int)
    for sol in verification_solutions:
        verification_key_counts[sol["unique_key"]] += 1
    
    # Find collisions (keys that appear more than once)
    collision_errors = []
    for key, count in verification_key_counts.items():
        if count > 1:
            collision_errors.append({
                "unique_key": key,
                "count": count,
                "error": f"Key '{key}' appears {count} times in verification solutions"
            })
    
    # Check for no matches (verification solutions without corresponding rollout data)
    rollout_responses = {item["response"].strip() for item in full_raw_rollout_data_array}
    no_matches_array = []
    for sol in verification_solutions:
        if sol["unique_key"] not in rollout_responses:
            no_matches_array.append({
                "unique_key": sol["unique_key"],
                "custom_id": sol["custom_id"]
            })
    
    has_collisions = len(collision_errors) > 0
    has_no_matches = len(no_matches_array) > 0
    
    return collision_errors, no_matches_array, has_collisions, has_no_matches


def validate_verification_value(verification_sol: Dict, model_name: str) -> Optional[Dict]:
    """
    Validate the isVerified value for a verification solution.
    
    Args:
        verification_sol: Verification solution dict
        model_name: Name of the model (e.g., "o4-mini")
        
    Returns:
        Dict with invalid value info if invalid, None if valid
    """
    # The field might be named differently in the actual data
    # Try both patterns: "{model_name}_isVerified" and just "isVerified"
    is_verified_value = None
    is_verified_key = None
    
    # First try the model-specific key
    model_specific_key = f"{model_name}_isVerified"
    if model_specific_key in verification_sol:
        is_verified_key = model_specific_key
        is_verified_value = verification_sol[model_specific_key]
    # Then try generic key
    elif "isVerified" in verification_sol:
        is_verified_key = "isVerified"
        is_verified_value = verification_sol["isVerified"]
    # For o4-mini, also check the exact field name from the original code
    elif model_name == "o4-mini" and "o4-mini_isVerified" in verification_sol:
        is_verified_key = "o4-mini_isVerified"
        is_verified_value = verification_sol["o4-mini_isVerified"]
    
    if is_verified_value not in [True, False, None]:
        return {
            "verification_custom_id": verification_sol.get("custom_id"),
            "verification_response": verification_sol.get("verification_response"),
            "isVerified_value": is_verified_value,
            "isVerified_key": is_verified_key,
            "model": model_name,
            "type": type(is_verified_value).__name__
        }
    return None


def merge_single_model_verification(rollout_item: Dict, 
                                  verification_lookup: Dict[str, Dict],
                                  model_name: str,
                                  stats: Dict[str, int],
                                  invalid_values_list: List[Dict]) -> Dict:
    """
    Merge verification data from a single model into the rollout item.
    
    Args:
        rollout_item: Single rollout data item
        verification_lookup: Lookup dict for verification solutions
        model_name: Name of the model
        stats: Statistics dictionary to update
        invalid_values_list: List to append invalid values to
        
    Returns:
        Dict with merged fields for this model
    """
    response_key = rollout_item["response"].strip()
    merged_fields = {}
    
    if response_key in verification_lookup:
        # Found matching verification solution
        verification_sol = verification_lookup[response_key]
        
        # Validate isVerified value
        invalid_value = validate_verification_value(verification_sol, model_name)
        if invalid_value:
            invalid_values_list.append(invalid_value)
        
        # Extract the isVerified value using flexible key matching
        is_verified_value = None
        # Try model-specific key first
        for key in [f"{model_name}_isVerified", "isVerified", "o4-mini_isVerified"]:
            if key in verification_sol:
                is_verified_value = verification_sol[key]
                break
        
        merged_fields[f"{model_name}_verification_custom_id"] = verification_sol.get("custom_id")
        merged_fields[f"{model_name}_verification_solution"] = verification_sol.get("verification_response")
        merged_fields[f"{model_name}_isVerified"] = is_verified_value
        
        stats[f"{model_name}_with_verification"] += 1
    else:
        # No matching verification solution found
        merged_fields[f"{model_name}_verification_custom_id"] = None
        merged_fields[f"{model_name}_verification_solution"] = None
        merged_fields[f"{model_name}_isVerified"] = None
        
        stats[f"{model_name}_without_verification"] += 1
    
    return merged_fields


def merge_rollout_with_multiple_verifications(
    full_raw_rollout_data_array: List[Dict],
    verification_solutions_dict: Dict[str, List[Dict]],
    output_path: str,
    check_collisions: bool = True
) -> List[Dict]:
    """
    Merge rollout data with multiple verification solution sets from different models.
    
    Args:
        full_raw_rollout_data_array: List of rollout data dicts (reference dataset)
        verification_solutions_dict: Dict mapping model names to their verification solutions
                                   e.g., {"o4-mini": [...], "gpt-4.1-mini": [...], "gpt-4.1-nano": [...]}
        output_path: Path to save the merged output file
        check_collisions: Whether to check for collisions before merging
        
    Returns:
        List of merged data items
    """
    # Check for collisions in each verification set if requested
    if check_collisions:
        for model_name, verification_solutions in verification_solutions_dict.items():
            collision_errors, no_matches, has_collisions, has_no_matches = check_for_collisions(
                verification_solutions, full_raw_rollout_data_array
            )
            
            if has_collisions:
                raise ValueError(f"{model_name}: {len(collision_errors)} collision errors found. Cannot proceed with merge.")
            
            if has_no_matches:
                print(f"⚠️  {model_name}: {len(no_matches)} verification solutions have no matching rollout data")
    
    # Create lookup dictionaries for each model's verification solutions
    verification_lookups = {}
    for model_name, verification_solutions in verification_solutions_dict.items():
        verification_lookups[model_name] = {sol["unique_key"]: sol for sol in verification_solutions}
    
    # Initialize statistics
    stats = defaultdict(int)
    invalid_verification_values = []
    
    # Merge the data - iterate over rollout data as reference
    merged_data = []
    for rollout_item in full_raw_rollout_data_array:
        # Start with base rollout fields
        merged_item = {
            "response_uid": rollout_item["response_uid"],
            "rollout_response": rollout_item["response"],
            "rollout_image_path": rollout_item["image_path"]
        }
        
        # Merge verification data from each model
        for model_name, verification_lookup in verification_lookups.items():
            model_fields = merge_single_model_verification(
                rollout_item, 
                verification_lookup, 
                model_name,
                stats,
                invalid_verification_values
            )
            merged_item.update(model_fields)
        
        merged_data.append(merged_item)
    
    # Save to file
    with open(output_path, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')
    
    # Print summary
    print(f"\n✅ Successfully merged {len(merged_data)} items to {output_path}")
    print(f"📊 Summary:")
    print(f"   - Total rollouts: {len(merged_data)}")
    
    for model_name in verification_solutions_dict.keys():
        print(f"\n   {model_name}:")
        print(f"     - With verification: {stats[f'{model_name}_with_verification']}")
        print(f"     - Without verification: {stats[f'{model_name}_without_verification']}")
    
    # Report invalid verification values by model
    if invalid_verification_values:
        print(f"\n⚠️  INVALID VERIFICATION VALUES FOUND: {len(invalid_verification_values)} items total")
        
        # Group by model
        invalid_by_model = defaultdict(list)
        for item in invalid_verification_values:
            invalid_by_model[item['model']].append(item)
        
        for model_name, model_invalids in invalid_by_model.items():
            print(f"\n   {model_name} ({len(model_invalids)} invalid values):")
            for item in model_invalids[:5]:  # Show first 5
                print(f"     - custom_id: {item['verification_custom_id']}, "
                      f"value: {item['isVerified_value']} (type: {item['type']})")
            if len(model_invalids) > 5:
                print(f"     ... and {len(model_invalids) - 5} more")
    else:
        print(f"\n✅ All verification values are valid (True/False/None)")
    
    return merged_data


def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load a JSONL file and return list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def process_dataset(dataset_name: str, 
                   base_dir: str,
                   model_names: List[str],
                   verification_dir: str = "verification_files",
                   rollout_dir: str = "flattened_rollout_files",
                   output_dir: str = "processed_full_verification_files") -> List[Dict]:
    """
    Process a single dataset by merging its rollout data with verification solutions from multiple models.
    
    Args:
        dataset_name: Name of the dataset (e.g., "AI2D")
        base_dir: Base directory containing all data
        model_names: List of model names to process
        verification_dir: Subdirectory containing verification files
        rollout_dir: Subdirectory containing rollout files
        output_dir: Subdirectory for output files
        
    Returns:
        List of merged data items
    """
    # Construct paths
    rollout_file = os.path.join(base_dir, rollout_dir, f"{dataset_name}_flattened.jsonl")
    
    # Load rollout data
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Loading rollout data from: {rollout_file}")
    
    if not os.path.exists(rollout_file):
        print(f"❌ Rollout file not found: {rollout_file}")
        return []
    
    full_raw_rollout_data_array = load_jsonl_file(rollout_file)
    print(f"✅ Loaded {len(full_raw_rollout_data_array)} rollout items")
    
    # Load verification solutions for each model
    verification_solutions_dict = {}
    for model_name in model_names:
        verification_file = os.path.join(
            base_dir, 
            verification_dir, 
            f"{dataset_name}_{model_name}_verification.jsonl"
        )
        
        if os.path.exists(verification_file):
            print(f"\nLoading {model_name} verification data...")
            verification_solutions_dict[model_name] = load_jsonl_file(verification_file)
            print(f"  ✅ Loaded {len(verification_solutions_dict[model_name])} verification items")
        else:
            print(f"  ⚠️  {model_name} verification file not found: {verification_file}")
            # Continue with empty list for this model
            verification_solutions_dict[model_name] = []
    
    # Create output directory if it doesn't exist
    output_path_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_path_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(
        output_path_dir, 
        f"{dataset_name}_final_all_models_merged.jsonl"
    )
    
    # Perform the merge
    try:
        merged_data = merge_rollout_with_multiple_verifications(
            full_raw_rollout_data_array,
            verification_solutions_dict,
            output_path,
            check_collisions=True
        )
        print(f"\n✨ Successfully processed {dataset_name}! Output saved to: {output_path}")
        return merged_data
    except Exception as e:
        print(f"\n❌ Error processing {dataset_name}: {str(e)}")
        return []


def process_multiple_datasets(dataset_names: List[str],
                            base_dir: str,
                            model_names: List[str],
                            **kwargs) -> Dict[str, List[Dict]]:
    """
    Process multiple datasets sequentially.
    
    Args:
        dataset_names: List of dataset names to process
        base_dir: Base directory containing all data
        model_names: List of model names to process
        **kwargs: Additional arguments to pass to process_dataset
        
    Returns:
        Dict mapping dataset names to their merged data
    """
    results = {}
    
    print(f"\n🚀 Starting batch processing of {len(dataset_names)} datasets")
    print(f"   Models: {', '.join(model_names)}")
    
    for dataset_name in dataset_names:
        merged_data = process_dataset(
            dataset_name, 
            base_dir, 
            model_names,
            **kwargs
        )
        results[dataset_name] = merged_data
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    successful = sum(1 for data in results.values() if len(data) > 0)
    print(f"✅ Successfully processed: {successful}/{len(dataset_names)} datasets")
    
    for dataset_name, data in results.items():
        if len(data) > 0:
            print(f"   - {dataset_name}: {len(data)} items")
        else:
            print(f"   - {dataset_name}: ❌ Failed")
    
    return results


def main():
    """Example usage of the merge function."""
    # Define paths
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    rollout_file = os.path.join(base_dir, "flattened_rollout_files/AI2D_flattened.jsonl")
    
    # Load rollout data
    print("Loading rollout data...")
    full_raw_rollout_data_array = load_jsonl_file(rollout_file)
    print(f"Loaded {len(full_raw_rollout_data_array)} rollout items")
    
    # Define verification files for each model
    verification_files = {
        "o4-mini": os.path.join(base_dir, "verification_files/AI2D_o4-mini_verification.jsonl"),
        "gpt-4.1-mini": os.path.join(base_dir, "verification_files/AI2D_gpt-4.1-mini_verification.jsonl"),
        "gpt-4.1-nano": os.path.join(base_dir, "verification_files/AI2D_gpt-4.1-nano_verification.jsonl")
    }
    
    # Load verification solutions for each model
    verification_solutions_dict = {}
    for model_name, filepath in verification_files.items():
        if os.path.exists(filepath):
            print(f"Loading {model_name} verification data...")
            verification_solutions_dict[model_name] = load_jsonl_file(filepath)
            print(f"  Loaded {len(verification_solutions_dict[model_name])} verification items")
        else:
            print(f"⚠️  {model_name} verification file not found: {filepath}")
    
    # Define output path
    output_path = os.path.join(
        base_dir, 
        "processed_full_verification_files/AI2D_final_all_models_merged.jsonl"
    )
    
    # Perform the merge
    merged_data = merge_rollout_with_multiple_verifications(
        full_raw_rollout_data_array,
        verification_solutions_dict,
        output_path,
        check_collisions=True
    )
    
    print(f"\n✨ Merge complete! Output saved to: {output_path}")


def example_batch_processing():
    """Example of processing multiple datasets."""
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    
    # Define datasets to process
    dataset_names = ["AI2D", "ChartQA", "DocVQA", "InfographicVQA"]
    
    # Define models to include
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    # Process all datasets
    results = process_multiple_datasets(
        dataset_names,
        base_dir,
        model_names
    )
    
    return results


def test_single_dataset():
    """Test processing a single dataset."""
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    dataset_name = "AI2D"
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    print("🧪 Testing single dataset processing...")
    merged_data = process_dataset(
        dataset_name,
        base_dir,
        model_names
    )
    
    if merged_data:
        print(f"\n📋 Sample merged item structure:")
        print(json.dumps(merged_data[0], indent=2))
    
    return merged_data


def verify_merge_output(merged_file_path: str):
    """Verify the structure and content of merged output file."""
    print(f"\n🔍 Verifying merged output: {merged_file_path}")
    
    if not os.path.exists(merged_file_path):
        print("❌ File not found!")
        return
    
    # Load and analyze the data
    data = load_jsonl_file(merged_file_path)
    print(f"✅ Loaded {len(data)} items")
    
    if not data:
        return
    
    # Analyze first item structure
    first_item = data[0]
    print("\n📊 Field structure:")
    for key in sorted(first_item.keys()):
        value_type = type(first_item[key]).__name__
        print(f"   - {key}: {value_type}")
    
    # Count verification coverage for each model
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    coverage_stats = {model: {"verified": 0, "not_verified": 0, "no_data": 0} 
                     for model in model_names}
    
    for item in data:
        for model in model_names:
            custom_id = item.get(f"{model}_verification_custom_id")
            is_verified = item.get(f"{model}_isVerified")
            
            if custom_id is None:
                coverage_stats[model]["no_data"] += 1
            elif is_verified is True:
                coverage_stats[model]["verified"] += 1
            elif is_verified is False:
                coverage_stats[model]["not_verified"] += 1
            else:
                coverage_stats[model]["no_data"] += 1
    
    print("\n📈 Verification coverage by model:")
    for model, stats in coverage_stats.items():
        total_with_data = stats["verified"] + stats["not_verified"]
        print(f"\n   {model}:")
        print(f"     - Items with verification: {total_with_data}")
        print(f"     - Verified as correct: {stats['verified']}")
        print(f"     - Verified as incorrect: {stats['not_verified']}")
        print(f"     - No verification data: {stats['no_data']}")
        if total_with_data > 0:
            accuracy = (stats["verified"] / total_with_data) * 100
            print(f"     - Verification accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run test on single dataset
            test_single_dataset()
        elif sys.argv[1] == "batch":
            # Run batch processing
            example_batch_processing()
        elif sys.argv[1] == "verify":
            # Verify a specific output file
            if len(sys.argv) > 2:
                verify_merge_output(sys.argv[2])
            else:
                print("Usage: python script.py verify <path_to_merged_file>")
    else:
        # Run default main function
        main() 