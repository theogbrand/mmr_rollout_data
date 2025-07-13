import json
import os
from pathlib import Path
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

def setup_logging(log_file: str = "verification_processing.log") -> logging.Logger:
    """Set up logging configuration with both file and console handlers."""
    logger = logging.getLogger("verification_processor")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_jsonl_to_dict(file_path: str) -> Dict[str, Any]:
    """Load JSONL file into a dictionary keyed by custom_id."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data[item['custom_id']] = item
    return data


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return the data."""
    with open(file_path, 'r') as f:
        return json.load(f)


def validate_custom_ids(jsonl_data: Dict[str, Any], json_data: List[Dict[str, Any]], logger: logging.Logger) -> bool:
    """Validate that custom_ids match between JSONL and JSON data."""
    json_ids = {item['custom_id'] for item in json_data} # we take the verification results as the reference since that is our limiting factor
    jsonl_ids = set(jsonl_data.keys())
    
    logger.info(f"number of custom_ids in verification result JSON: {len(json_ids)}")
    logger.info(f"number of custom_ids in Queries JSONL: {len(jsonl_ids)}")

    if jsonl_ids != json_ids:
        missing_in_json = jsonl_ids - json_ids
        missing_in_jsonl = json_ids - jsonl_ids
        if missing_in_json:
            logger.warning(f"custom_ids in Queries JSONL sent but not found in verification result JSON: {missing_in_json}")
            logger.warning(f"number of missing verification results: {len(missing_in_json)}")
        if missing_in_jsonl: # should not actually reach here
            logger.warning(f"custom_ids in verification result JSON sent but not found in Queries JSONL: {missing_in_jsonl}")
            logger.warning(f"number of missing verification queries: {len(missing_in_jsonl)}")
        return False
    else:
        logger.info(f"All custom_ids match")
        return True


def merge_and_save_data(query_jsonl_data: Dict[str, Any], verification_result_json_data: List[Dict[str, Any]], output_path: str, logger: logging.Logger) -> int:
    """Merge JSONL and JSON data and save to file."""
    with open(output_path, 'w') as f:
        for item in verification_result_json_data:
            merged = {**query_jsonl_data[item['custom_id']], **item} # query is the base file. Merge is skipped if custom_id (intersection ID) not found in verification result JSON, the "verification_response" key will just not exist in the resulting merged item
            f.write(json.dumps(merged) + '\n')
    
        logger.info(f"Merged {len(verification_result_json_data)} items to {output_path}") # 
    return len(verification_result_json_data)


def extract_conclusion(verification_response: str) -> Optional[str]:
    """Extract conclusion text from verification response."""
    conclusion_pattern = re.compile(r'<conclusion>(.*?)</conclusion>', re.DOTALL)
    match = conclusion_pattern.search(verification_response)
    return match.group(1).strip() if match else None


def process_verification_results(merged_file_path: str, model_name: str, logger: logging.Logger) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Process verification results and add isVerified flags."""
    processed_data = []
    stats = {"correct": 0, "incorrect": 0, "invalid": 0}
    
    with open(merged_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line)
            
            verification_response = item.get("verification_response", "custom_id query has no corresponding verification response")
            conclusion_pattern = re.compile(r'<conclusion>(.*?)</conclusion>', re.DOTALL)
            conclusion_match = conclusion_pattern.search(verification_response)
            
            if conclusion_match:
                conclusion_text = conclusion_match.group(1).strip()
                if conclusion_text.lower() == "correct":
                    item[f"{model_name}_isVerified"] = True
                    stats["correct"] += 1
                elif conclusion_text.lower() == "incorrect":
                    item[f"{model_name}_isVerified"] = False
                    stats["incorrect"] += 1
                else:
                    logger.warning(f"Invalid conclusion text for custom_id: {item['custom_id']}: '{conclusion_text}'")
                    stats["invalid"] += 1
                    item[f"{model_name}_isVerified"] = None
            else:
                logger.warning(f"No <conclusion> tags found for custom_id: {item['custom_id']}: {verification_response}")
                stats["invalid"] += 1
                item[f"{model_name}_isVerified"] = None
            
            processed_data.append(item)
    
    return processed_data, stats


def save_processed_data(processed_data: List[Dict[str, Any]], output_path: str, logger: logging.Logger):
    """Save processed data to JSONL file."""
    with open(output_path, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved processed data to: {output_path}")


def process_model_verification(model: str, dataset_name: str, dataset_subset_name: str, base_dir: str, output_dir: str, logger: logging.Logger):
    """Process verification files for a single model."""
    logger.info(f"\nProcessing model: {model} and dataset subset: {dataset_subset_name}")
    if dataset_subset_name:
        dataset_subset_name_str = f"_{dataset_subset_name}"
    else:
        dataset_subset_name_str = ""
    
    # File paths
    batch_query_jsonl_file = f"{base_dir}/flattened_verification_query_files/{dataset_name}_{model}{dataset_subset_name_str}_verification_query_flattened.jsonl"
    batch_verification_result_json_file = f"{base_dir}/flattened_verification_result_files/{dataset_name}_{model}{dataset_subset_name_str}_verification_result_flattened.json"

    merged_output_path = os.path.join(output_dir, f"{dataset_name}_{model}{dataset_subset_name_str}_verification_merged.jsonl")
    final_output_path = f"{output_dir}/{dataset_name}_final_verification_processed_{model}{dataset_subset_name_str}.jsonl"
    
    # Load data
    batch_query_data = load_jsonl_to_dict(batch_query_jsonl_file)
    batch_verification_data = load_json_file(batch_verification_result_json_file)
    
    # Validate and merge
    validate_custom_ids(batch_query_data, batch_verification_data, logger)
    merge_and_save_data(batch_query_data, batch_verification_data, merged_output_path, logger)
    
    # Process verification results
    processed_data, stats = process_verification_results(merged_output_path, model, logger)
    
    # Log statistics
    logger.info(f"Processing complete:")
    logger.info(f"  Correct: {stats['correct']}")
    logger.info(f"  Incorrect: {stats['incorrect']}")
    logger.info(f"  Invalid: {stats['invalid']}")
    logger.info(f"  Total: {len(processed_data)}")
    
    # Save processed data
    save_processed_data(processed_data, final_output_path, logger)


def main():
    """Main function to process all models."""
    # Configuration
    models = ["gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]
   
    dataset_subset_split_mapping = {
        "RAVEN": ["center_single", "distribute_four", "distribute_nine", "in_center_single_out_center_single_train", "in_distribute_four_out_center_single_train", "left_center_single_right_center_single_train", "up_center_single_down_center_single_train"],
        "CLEVR": ["CLEVR_first_5k", "CLEVR_second_5k"],
        "dvqa": ["dvqa_first_5k", "dvqa_second_5k", "dvqa_third_5k"]
    } 
    
    dataset_name = "dvqa" # TODO: Set this before running
     
    base_dir = "/mnt/fast10/brandon/mmr_rollout_data"
    output_dir = "/mnt/fast10/brandon/mmr_rollout_data/merged_verification_files"
    
    # Create logs directory
    logs_dir = os.path.join(base_dir, "process_verification_files_logs")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"verification_processing_{timestamp}.log")
    logger = setup_logging(log_file)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting verification processing for all models")
    logger.info(f"Models to process: {models}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process each model
    for model in models:
        if dataset_subset_split_mapping[dataset_name]:
            for dataset_subset_name in dataset_subset_split_mapping[dataset_name]:
                process_model_verification(model, dataset_name, dataset_subset_name, base_dir, output_dir, logger)
        else:
            process_model_verification(model, dataset_name, None, base_dir, output_dir, logger)
    
    logger.info("Verification processing completed for all models")


if __name__ == "__main__":
    main() 
# dataset_name = "CLEVR" # TODO: Set this before running
# usage: python process_verification_files.py