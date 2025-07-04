import json
import os
import logging
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def get_safe_model_name(model_name: str) -> str:
    """Convert model name to safe format for column names by replacing hyphens with underscores."""
    return model_name.replace("-", "_")


def setup_logger(log_dir: str = "merge_verification_and_rollout_logs", 
                log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('merge_verification_rollout')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"merge_verification_rollout_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_filepath}")
    
    return logger


def check_for_collisions(verification_solutions: List[Dict], 
                         full_raw_rollout_data_array: List[Dict],
                         logger: logging.Logger) -> Tuple[List, List, bool, bool]:
    """
    Check for collision errors between verification solutions and rollout data.
    
    Args:
        verification_solutions: List of verification solution dicts
        full_raw_rollout_data_array: List of rollout data dicts
        logger: Logger instance
        
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
    
    if has_collisions:
        logger.error(f"Found {len(collision_errors)} collision errors")
        for error in collision_errors:
            logger.error(f"  - {error['error']}")
    
    if has_no_matches:
        logger.warning(f"Found {len(no_matches_array)} verification solutions with no matching rollout data")
    
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
    
    # Get safe model name for consistent field naming
    safe_model_name = get_safe_model_name(model_name)
    
    # First try the safe model-specific key
    safe_model_key = f"{safe_model_name}_isVerified"
    if safe_model_key in verification_sol:
        is_verified_key = safe_model_key
        is_verified_value = verification_sol[safe_model_key]
    
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
        safe_model_name = get_safe_model_name(model_name)
        is_verified_value = verification_sol[f"{safe_model_name}_isVerified"] # in the extract_verification_solutions function, we already use safe_model_name to store the isVerified value instead of model_name
        
        merged_fields[f"{safe_model_name}_verification_custom_id"] = verification_sol.get(f"{safe_model_name}_custom_id", f"ERROR: {model_name} custom_id not found at merge_single_model_verification step")
        merged_fields[f"{safe_model_name}_verification_solution"] = verification_sol.get(f"{safe_model_name}_verification_response", f"ERROR: {model_name} verification_response not found at merge_single_model_verification step")
        merged_fields[f"{safe_model_name}_isVerified"] = is_verified_value
        
        stats[f"{safe_model_name}_with_verification"] += 1
    else:
        # No matching verification solution found
        safe_model_name = get_safe_model_name(model_name)
        
        merged_fields[f"{safe_model_name}_verification_custom_id"] = None
        merged_fields[f"{safe_model_name}_verification_solution"] = None
        merged_fields[f"{safe_model_name}_isVerified"] = None
        
        stats[f"{safe_model_name}_without_verification"] += 1
    
    return merged_fields


def merge_rollout_with_multiple_verifications(
    full_raw_rollout_data_array: List[Dict],
    verification_solutions_dict: Dict[str, List[Dict]],
    output_path: str,
    logger: logging.Logger,
    check_collisions: bool = True
) -> List[Dict]:
    """
    Merge rollout data with multiple verification solution sets from different models.
    
    Args:
        full_raw_rollout_data_array: List of rollout data dicts (reference dataset)
        verification_solutions_dict: Dict mapping model names to their verification solutions
                                   e.g., {"o4-mini": [...], "gpt-4.1-mini": [...], "gpt-4.1-nano": [...]}
        output_path: Path to save the merged output file
        logger: Logger instance
        check_collisions: Whether to check for collisions before merging
        
    Returns:
        List of merged data items
    """
    # Check for collisions in each verification set if requested
    if check_collisions:
        for model_name, verification_solutions in verification_solutions_dict.items():
            logger.info(f"Checking collisions for {model_name}...")
            collision_errors, no_matches, has_collisions, has_no_matches = check_for_collisions(
                verification_solutions, full_raw_rollout_data_array, logger
            )
            
            if has_collisions:
                raise ValueError(f"{model_name}: {len(collision_errors)} collision errors found. Cannot proceed with merge.")
            
            if has_no_matches:
                logger.warning(f"{model_name}: {len(no_matches)} verification solutions have no matching rollout data")
    
    # Create lookup dictionaries for each model's verification solutions
    verification_lookups = {}
    for model_name, verification_solutions in verification_solutions_dict.items():
        verification_lookups[model_name] = {sol["unique_key"]: sol for sol in verification_solutions}
        logger.info(f"Created lookup for {model_name}: {len(verification_solutions)} items")
    
    # Initialize statistics
    stats = defaultdict(int)
    invalid_verification_values = []
    
    # Merge the data - iterate over rollout data as reference
    logger.info(f"Starting merge of {len(full_raw_rollout_data_array)} rollout items...")
    merged_data = []
    for i, rollout_item in enumerate(full_raw_rollout_data_array):
        # Start with base rollout fields
        merged_item = {
            "response_uid": rollout_item["uid"],
            "rollout_question": rollout_item["question"],
            "rollout_response": rollout_item["response"],
            "rollout_answer": rollout_item["answer"],
            "rollout_steps_with_score": rollout_item["steps_with_score"],
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
        
        # Log progress every 1000 items
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(full_raw_rollout_data_array)} items")
    
    # Save to file
    logger.info(f"Saving merged data to {output_path}")
    with open(output_path, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')
    
    # Print summary
    logger.info(f"Successfully merged {len(merged_data)} items to {output_path}")
    logger.info("Summary:")
    logger.info(f"   - Total rollouts: {len(merged_data)}")
    
    for model_name in verification_solutions_dict.keys():
        safe_model_name = get_safe_model_name(model_name)
        logger.info(f"\n   {model_name}:")
        logger.info(f"     - With verification: {stats[f'{safe_model_name}_with_verification']}")
        logger.info(f"     - Without verification: {stats[f'{safe_model_name}_without_verification']}")
    
    # Report invalid verification values by model
    if invalid_verification_values:
        logger.warning(f"INVALID VERIFICATION VALUES FOUND: {len(invalid_verification_values)} items total")
        
        # Group by model
        invalid_by_model = defaultdict(list)
        for item in invalid_verification_values:
            invalid_by_model[item['model']].append(item)
        
        for model_name, model_invalids in invalid_by_model.items():
            logger.warning(f"\n   {model_name} ({len(model_invalids)} invalid values):")
            for item in model_invalids[:5]:  # Show first 5
                logger.warning(f"     - custom_id: {item['verification_custom_id']}, "
                              f"value: {item['isVerified_value']} (type: {item['type']})")
            if len(model_invalids) > 5:
                logger.warning(f"     ... and {len(model_invalids) - 5} more")
    else:
        logger.info("All verification values are valid (True/False/None)")
    
    return merged_data


def load_jsonl_file(filepath: str, logger: logging.Logger) -> List[Dict]:
    """Load a JSONL file and return list of dicts."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num} in {filepath}: {e}")
                    continue
        logger.info(f"Successfully loaded {len(data)} items from {filepath}")
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        raise
    
    return data


def extract_verification_solutions(merged_verification_file: str, model_name: str, logger: logging.Logger) -> List[Dict]:
    """
    Extract verification solutions from a verification file.
    """
    verification_solutions = []
    single_tag_custom_ids = []
    no_tag_custom_ids = []
    solution_pattern = re.compile(r'<solution>(.*?)</solution>', re.DOTALL) # we are using the solution sent to be verified as the intersection key (see README.md for more details)
    safe_model_name = get_safe_model_name(model_name)
    with open(merged_verification_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line)
            try:
                text = item["body"]["messages"][0]["content"][0]["text"] # this is from verification query request, not the verification response
                # Find all matches and get the second one because the first one is the prompt question, the second one is the actual solution
                matches = solution_pattern.findall(text)
                if len(matches) >= 2:
                    solution_text = matches[1].strip()  # Get second occurrence
                    if solution_text:  # Only add non-empty verification_solutions 
                        verification_solutions.append({
                            f"{safe_model_name}_custom_id": item.get("custom_id", f"ERROR: {model_name} custom_id not found"),
                            "unique_key": solution_text, # using the solution as the intersection key
                            f"{safe_model_name}_verification_response": item.get(f"verification_response", f"ERROR: {model_name} verification_response not found"),
                            f"{safe_model_name}_isVerified": item.get(f"{model_name}_isVerified", f"ERROR: {model_name} isVerified not found")
                        })
                elif len(matches) == 1:
                    custom_id = item.get("custom_id", f"ERROR: {model_name} custom_id not found")
                    single_tag_custom_ids.append(custom_id)
                    logger.warning(f"Warning: Only one <solution> tag found in line {line_num} for {model_name}")
                else:
                    custom_id = item.get("custom_id", f"ERROR: {model_name} custom_id not found")
                    no_tag_custom_ids.append(custom_id)
                    logger.warning(f"Warning: No <solution> tags found in line {line_num} for {model_name}")
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Error accessing text in line {line_num}: {e}")

        logger.info(f"Extracted {len(verification_solutions)} valid verification_solutions")
        logger.info(f"Found {len(single_tag_custom_ids)} items with only one solution tag: {single_tag_custom_ids}")
        logger.info(f"Found {len(no_tag_custom_ids)} items with no solution tags: {no_tag_custom_ids}")
        return verification_solutions


def process_dataset(dataset_name: str, 
                   base_dir: str,
                   model_names: List[str],
                   logger: logging.Logger,
                   verification_dir: str = "merged_verification_files",
                   rollout_dir: str = "flattened_rollout_files",
                   output_dir: str = "processed_full_verification_files") -> List[Dict]:
    """
    Process a single dataset by merging its rollout data with verification solutions from multiple models.
    
    Args:
        dataset_name: Name of the dataset (e.g., "AI2D")
        base_dir: Base directory containing all data
        model_names: List of model names to process
        logger: Logger instance
        verification_dir: Subdirectory containing verification files
        rollout_dir: Subdirectory containing rollout files
        output_dir: Subdirectory for output files
        
    Returns:
        List of merged data items
    """
    # Construct paths
    rollout_file = os.path.join(base_dir, rollout_dir, f"{dataset_name}_flattened.jsonl")
    
    # Load rollout data
    logger.info("="*60)
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info("="*60)
    logger.info(f"Loading rollout data from: {rollout_file}")
    
    if not os.path.exists(rollout_file):
        logger.error(f"Rollout file not found: {rollout_file}")
        return []
    
    full_raw_rollout_data_array = load_jsonl_file(rollout_file, logger)
    
    # Load verification solutions for each model
    verification_solutions_dict = {}
    for model_name in model_names:
        # safe_model_name = get_safe_model_name(model_name)
        verification_file = os.path.join(
            base_dir, 
            verification_dir, 
            f"{dataset_name}_final_verification_processed_{model_name}.jsonl"
        )
        
        if os.path.exists(verification_file):
            logger.info(f"\nLoading {model_name} verification data...")
            verification_solutions_dict[model_name] = extract_verification_solutions(verification_file, model_name, logger)
        else:
            logger.warning(f"{model_name} verification file not found: {verification_file}")
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
            logger,
            check_collisions=True
        )
        logger.info(f"Successfully processed {dataset_name}! Output saved to: {output_path}")
        return merged_data
    except Exception as e:
        logger.error(f"Error processing {dataset_name}: {str(e)}")
        return []


def process_multiple_datasets(dataset_names: List[str],
                            base_dir: str,
                            model_names: List[str],
                            logger: logging.Logger,
                            **kwargs) -> Dict[str, List[Dict]]:
    """
    Process multiple datasets sequentially.
    
    Args:
        dataset_names: List of dataset names to process
        base_dir: Base directory containing all data
        model_names: List of model names to process
        logger: Logger instance
        **kwargs: Additional arguments to pass to process_dataset
        
    Returns:
        Dict mapping dataset names to their merged data
    """
    results = {}
    
    logger.info("="*60)
    logger.info(f"Starting batch processing of {len(dataset_names)} datasets")
    logger.info(f"   Models: {', '.join(model_names)}")
    logger.info("="*60)
    
    for i, dataset_name in enumerate(dataset_names, 1):
        logger.info(f"\nProcessing dataset {i}/{len(dataset_names)}: {dataset_name}")
        merged_data = process_dataset(
            dataset_name, 
            base_dir, 
            model_names,
            logger,
            **kwargs
        )
        results[dataset_name] = merged_data
    
    # Summary
    logger.info("="*60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*60)
    successful = sum(1 for data in results.values() if len(data) > 0)
    logger.info(f"Successfully processed: {successful}/{len(dataset_names)} datasets")
    
    for dataset_name, data in results.items():
        if len(data) > 0:
            logger.info(f"   - {dataset_name}: {len(data)} items")
        else:
            logger.error(f"   - {dataset_name}: Failed")
    
    return results


def test_single_dataset():
    """Test processing a single dataset."""
    logger = setup_logger()
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    dataset_name = "AI2D"
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    logger.info("🧪 Testing single dataset processing...")
    merged_data = process_dataset(
        dataset_name,
        base_dir,
        model_names,
        logger
    )
    
    if merged_data:
        logger.info("📋 Sample merged item structure:")
        logger.info(json.dumps(merged_data[0], indent=2))
    
    return merged_data


def verify_merge_output(merged_file_path: str, logger: logging.Logger):
    """Verify the structure and content of merged output file."""
    logger.info(f"🔍 Verifying merged output: {merged_file_path}")
    
    if not os.path.exists(merged_file_path):
        logger.error("File not found!")
        return
    
    # Load and analyze the data
    data = load_jsonl_file(merged_file_path, logger)
    
    if not data:
        return
    
    # Analyze first item structure
    first_item = data[0]
    logger.info("📊 Field structure:")
    for key in sorted(first_item.keys()):
        value_type = type(first_item[key]).__name__
        logger.info(f"   - {key}: {value_type}")
    
    # Count verification coverage for each model
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    coverage_stats = {model: {"verified": 0, "not_verified": 0, "no_data": 0} 
                     for model in model_names}
    
    for item in data:
        for model in model_names:
            safe_model_name = get_safe_model_name(model)
            custom_id = item.get(f"{safe_model_name}_verification_custom_id")
            is_verified = item.get(f"{safe_model_name}_isVerified")
            
            if custom_id is None:
                coverage_stats[model]["no_data"] += 1
            elif is_verified is True:
                coverage_stats[model]["verified"] += 1
            elif is_verified is False:
                coverage_stats[model]["not_verified"] += 1
            else:
                coverage_stats[model]["no_data"] += 1
    
    logger.info("📈 Verification coverage by model:")
    for model, stats in coverage_stats.items():
        total_with_data = stats["verified"] + stats["not_verified"]
        logger.info(f"\n   {model}:")
        logger.info(f"     - Items with verification: {total_with_data}")
        logger.info(f"     - Verified as correct: {stats['verified']}")
        logger.info(f"     - Verified as incorrect: {stats['not_verified']}")
        logger.info(f"     - No verification data: {stats['no_data']}")
        if total_with_data > 0:
            accuracy = (stats["verified"] / total_with_data) * 100
            logger.info(f"     - Verification accuracy: {accuracy:.2f}%")


def example_batch_processing():
    """Example of processing multiple datasets."""
    logger = setup_logger()
    
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    
    # Define datasets to process
    dataset_names = ["AI2D", "ChartQA", "DocVQA", "InfographicVQA"]
    
    # Define models to include
    model_names = ["o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    # Process all datasets
    results = process_multiple_datasets(
        dataset_names,
        base_dir,
        model_names,
        logger
    )
    
    return results


def main():
    """Example usage of the merge function."""
    logger = setup_logger()
    
    # Define paths
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    rollout_file = os.path.join(base_dir, "flattened_rollout_files/AI2D_flattened.jsonl")
    
    # Load rollout data
    logger.info("Loading rollout data...")
    full_raw_rollout_data_array = load_jsonl_file(rollout_file, logger)
    
    # Define verification files for each model
    verification_files = {
        "o4-mini": os.path.join(base_dir, "merged_verification_files/AI2D_o4-mini_final_verification_processed_o4-mini.jsonl"),
        "gpt-4.1-mini": os.path.join(base_dir, "merged_verification_files/AI2D_gpt-4.1-mini_final_verification_processed_gpt-4.1-mini.jsonl"),
        "gpt-4.1-nano": os.path.join(base_dir, "merged_verification_files/AI2D_gpt-4.1-nano_final_verification_processed_gpt-4.1-nano.jsonl")
    }
    
    # Load verification solutions for each model
    verification_solutions_dict = {}
    for model_name, filepath in verification_files.items():
        if os.path.exists(filepath):
            logger.info(f"Loading {model_name} verification data...")
            verification_solutions_dict[model_name] = load_jsonl_file(filepath, logger) # every file is stored as a list of dictionaries containing the verification solutions as a single item
        else:
            logger.warning(f"{model_name} verification file not found: {filepath}")
    
    # Define output path
    output_path = os.path.join(
        base_dir, 
        "processed_full_verification_files/AI2D_final_all_models_merged.jsonl"
    )
    
    # Perform the merge
    merged_data = merge_rollout_with_multiple_verifications(
        full_raw_rollout_data_array,
        verification_solutions_dict, # contains the verification solutions for each model
        output_path,
        logger,
        check_collisions=True
    )
    
    logger.info(f"Merge complete! Output saved to: {output_path}")


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
                logger = setup_logger()
                verify_merge_output(sys.argv[2], logger)
            else:
                print("Usage: python script.py verify <path_to_merged_file>")
    else:
        # Run default main function
        main() 