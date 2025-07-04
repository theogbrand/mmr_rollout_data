#!/usr/bin/env python3
"""
Example usage of the merge_rollout_and_verification_files module.
This script demonstrates how to merge rollout data with verification solutions from multiple models.
"""

from merge_rollout_and_verification_files import (
    process_dataset,
    process_multiple_datasets,
    verify_merge_output,
    load_jsonl_file,
    merge_rollout_with_multiple_verifications
)


def example_1_single_dataset():
    """Example 1: Process a single dataset with verification from 3 models."""
    print("=" * 80)
    print("EXAMPLE 1: Processing single dataset (AI2D)")
    print("=" * 80)
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    dataset_name = "AI2D"
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    # Process the dataset
    merged_data = process_dataset(
        dataset_name=dataset_name,
        base_dir=base_dir,
        model_names=model_names,
        verification_dir="verification_files",
        rollout_dir="flattened_rollout_files",
        output_dir="processed_full_verification_files"
    )
    
    # Verify the output
    output_file = f"{base_dir}/processed_full_verification_files/{dataset_name}_final_all_models_merged.jsonl"
    verify_merge_output(output_file)
    
    return merged_data


def example_2_batch_processing():
    """Example 2: Process multiple datasets in batch."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch processing multiple datasets")
    print("=" * 80)
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    
    # List of datasets to process
    dataset_names = [
        "AI2D",
        "ChartQA", 
        "DocVQA",
        "InfographicVQA",
        "TextVQA"
    ]
    
    # Models to include in merge
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    # Process all datasets
    results = process_multiple_datasets(
        dataset_names=dataset_names,
        base_dir=base_dir,
        model_names=model_names
    )
    
    return results


def example_3_custom_merge():
    """Example 3: Custom merge with manual file loading."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom merge with manual control")
    print("=" * 80)
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    
    # Load rollout data manually
    rollout_file = f"{base_dir}/flattened_rollout_files/AI2D_flattened.jsonl"
    rollout_data = load_jsonl_file(rollout_file)
    print(f"Loaded {len(rollout_data)} rollout items")
    
    # Load verification files manually
    verification_solutions_dict = {
        "o4-mini": load_jsonl_file(f"{base_dir}/verification_files/AI2D_o4-mini_verification.jsonl"),
        "gpt-4.1-mini": load_jsonl_file(f"{base_dir}/verification_files/AI2D_gpt-4.1-mini_verification.jsonl"),
        "gpt-4.1-nano": load_jsonl_file(f"{base_dir}/verification_files/AI2D_gpt-4.1-nano_verification.jsonl")
    }
    
    # Custom output path
    output_path = f"{base_dir}/custom_output/AI2D_custom_merged.jsonl"
    
    # Perform merge
    merged_data = merge_rollout_with_multiple_verifications(
        full_raw_rollout_data_array=rollout_data,
        verification_solutions_dict=verification_solutions_dict,
        output_path=output_path,
        check_collisions=True
    )
    
    return merged_data


def example_4_partial_models():
    """Example 4: Process with only some models available."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Processing with partial model coverage")
    print("=" * 80)
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    dataset_name = "AI2D"
    
    # Only process models that we have verification for
    # The function will handle missing files gracefully
    model_names = ["o4-mini", "gpt-4.1-mini"]  # Excluding gpt-4.1-nano
    
    merged_data = process_dataset(
        dataset_name=dataset_name,
        base_dir=base_dir,
        model_names=model_names
    )
    
    return merged_data


def example_5_analyze_results():
    """Example 5: Analyze merged results."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Analyzing merged results")
    print("=" * 80)
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    merged_file = f"{base_dir}/processed_full_verification_files/AI2D_final_all_models_merged.jsonl"
    
    # Load the merged data
    data = load_jsonl_file(merged_file)
    
    print(f"\nTotal items: {len(data)}")
    
    # Analyze agreement between models
    agreement_stats = {
        "all_agree_verified": 0,
        "all_agree_not_verified": 0,
        "disagreement": 0,
        "partial_data": 0
    }
    
    for item in data:
        verifications = []
        for model in ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]:
            is_verified = item.get(f"{model}_isVerified")
            if is_verified is not None:
                verifications.append(is_verified)
        
        if len(verifications) == 0:
            continue
        elif len(verifications) < 3:
            agreement_stats["partial_data"] += 1
        elif all(v is True for v in verifications):
            agreement_stats["all_agree_verified"] += 1
        elif all(v is False for v in verifications):
            agreement_stats["all_agree_not_verified"] += 1
        else:
            agreement_stats["disagreement"] += 1
    
    print("\n📊 Model Agreement Analysis:")
    for stat, count in agreement_stats.items():
        print(f"   - {stat}: {count}")
    
    # Find examples of disagreement
    print("\n🔍 Examples of model disagreement (first 3):")
    disagreement_count = 0
    for item in data:
        verifications = {}
        for model in ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]:
            is_verified = item.get(f"{model}_isVerified")
            if is_verified is not None:
                verifications[model] = is_verified
        
        if len(verifications) >= 2 and len(set(verifications.values())) > 1:
            disagreement_count += 1
            if disagreement_count <= 3:
                print(f"\n   Response UID: {item['response_uid']}")
                print(f"   Response: {item['rollout_response'][:100]}...")
                for model, verdict in verifications.items():
                    print(f"   - {model}: {verdict}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == "1":
            example_1_single_dataset()
        elif example_num == "2":
            example_2_batch_processing()
        elif example_num == "3":
            example_3_custom_merge()
        elif example_num == "4":
            example_4_partial_models()
        elif example_num == "5":
            example_5_analyze_results()
        else:
            print("Usage: python example_merge_usage.py [1|2|3|4|5]")
    else:
        # Run all examples
        print("Running all examples...")
        example_1_single_dataset()
        # example_2_batch_processing()  # Commented out to avoid processing all datasets
        # example_3_custom_merge()
        # example_4_partial_models()
        # example_5_analyze_results() 