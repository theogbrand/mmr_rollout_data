import json
import os
import re
from tqdm import tqdm

def validate_rollout_response(rollout_response):
    """
    Validate that rollout_response follows the required format:
    - [Visual Elements]|[Perception] section with valid step tags
    - [Reasoning] section with valid step tags  
    - <correct_answer></correct_answer> section
    """
    if not rollout_response:
        return False
    
    # Check for required sections
    has_visual_or_perception = ('[Visual Elements]' in rollout_response or 
                               '[Perception]' in rollout_response)
    has_reasoning = '[Reasoning]' in rollout_response
    has_correct_answer = '<correct_answer>' in rollout_response and '</correct_answer>' in rollout_response
    
    if not (has_visual_or_perception and has_reasoning and has_correct_answer):
        return False
    
    # Find section boundaries
    visual_start = rollout_response.find('[Visual Elements]')
    perception_start = rollout_response.find('[Perception]')
    
    # Use whichever section exists
    first_section_start = max(visual_start, perception_start)
    if first_section_start == -1:
        return False
    
    reasoning_start = rollout_response.find('[Reasoning]')
    answer_start = rollout_response.find('<correct_answer>')
    answer_end = rollout_response.find('</correct_answer>')
    
    if reasoning_start == -1 or answer_start == -1 or answer_end == -1:
        return False
    
    # Validate section order
    if not (first_section_start < reasoning_start < answer_start < answer_end):
        return False
    
    # Extract sections
    first_section = rollout_response[first_section_start:reasoning_start].strip()
    reasoning_section = rollout_response[reasoning_start:answer_start].strip()
    
    # Validate step tags in both sections
    def validate_step_tags(section_content):
        """Check that all step tags have proper opening and closing tags"""
        # Find all step tags
        step_pattern = r'<step_(\d+)>'
        opening_tags = re.findall(step_pattern, section_content)
        
        for step_num in opening_tags:
            opening_tag = f'<step_{step_num}>'
            closing_tag = f'</step_{step_num}>'
            
            # Check if both opening and closing tags exist
            if opening_tag not in section_content or closing_tag not in section_content:
                return False
            
            # Check that closing tag comes after opening tag
            opening_pos = section_content.find(opening_tag)
            closing_pos = section_content.find(closing_tag)
            
            if opening_pos == -1 or closing_pos == -1 or opening_pos >= closing_pos:
                return False
        
        return True
    
    # Validate step tags in both sections
    if not validate_step_tags(first_section) or not validate_step_tags(reasoning_section):
        return False
    
    return True

def find_problematic_uuids(data):
    """Find problematic rollout_uuids based on rollout_response format validation"""
    problematic_uuids = []
    
    for item in tqdm(data, desc="Validating rollouts"):
        rollout_uuid = item.get('rollout_uuid')
        rollout_response = item.get('rollout_response', '')
        
        # Check if rollout_response is valid
        if not validate_rollout_response(rollout_response):
            if rollout_uuid not in problematic_uuids:
                problematic_uuids.append(rollout_uuid)
    
    return problematic_uuids

def check_uuid_uniqueness(data, uuid):
    """Check that a UUID appears exactly once in the data"""
    count = sum(1 for item in data if item.get('rollout_uuid') == uuid)
    return count == 1

def drop_problematic_rollouts(data, problematic_uuids):
    """Drop (remove) problematic rollouts from the data"""
    original_count = len(data)
    
    # Filter out problematic rollouts
    filtered_data = []
    dropped_count = 0
    
    for item in tqdm(data, desc="Dropping problematic rollouts"):
        rollout_uuid = item.get('rollout_uuid')
        if rollout_uuid in problematic_uuids:
            # Verify uniqueness before dropping
            
            #TODO: Already checked for RAVEN. take long time so run separately and skip. Verify uniqueness before updating
            if not check_uuid_uniqueness(data, rollout_uuid):
                print(f"ERROR: UUID {rollout_uuid} appears multiple times in data - skipping drop")
                raise ValueError(f"ERROR: UUID {rollout_uuid} appears multiple times in data - skipping drop")

            print(f"Dropping rollout_uuid: {rollout_uuid}")
            print(f"item['rollout_steps_with_score']: {item['rollout_steps_with_score']}")
            print(f"item['rollout_response']: {item['rollout_response']}")
            dropped_count += 1
        else:
            filtered_data.append(item)
 
    print(f"Dropped {dropped_count} problematic rollouts out of {original_count} total")
    return filtered_data, dropped_count

def main():
    # Directory path
    data_dir = "/mnt/fast10/brandon/mmr_rollout_data/final_combined_MC_and_verification_files_updated_rollouts" # previously ran on _updated_rollouts which was a copy of the last 3 JSONL files that the "update verification value" was ran from
    
    dataset_files = [
        # TODO: redo drop and update steps in #4, then convert to prm again
        "InfoVQA_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "vqav2_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "CLEVR_final_mc_rollouts_with_all_models_verification_merged.jsonl",

        # TODO: in progress of converting to final state from source final_combined_MC_and_verification_files_updated_rollouts after drop and update steps in #4 completed
        # "RAVEN_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        # "dvqa_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        # "AI2D_final_mc_rollouts_with_all_models_verification_merged.jsonl"
    ]
    
    # Process each file
    for filename in dataset_files:
        print(f"\n{'='*60}")
        print(f"Processing file: {filename}")
        print(f"{'='*60}")
        
        filepath = os.path.join(data_dir, filename)
        
        # Load the JSONL file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data = [json.loads(line) for line in lines]
        print(f"Loaded {len(data)} items from {filename}")
        
        # Find problematic UUIDs
        problematic_uuids = find_problematic_uuids(data)
        print(f"Found {len(problematic_uuids)} problematic UUIDs")
        
        if problematic_uuids:
            print(f"Problematic UUIDs: {problematic_uuids}")
            
            # Check uniqueness for all problematic UUIDs
            all_unique = True
            # print(f"Skipping check_uuid_uniqueness (checked)")
            print(f"starting check_uuid_uniqueness")
            for uuid in tqdm(problematic_uuids, desc="Checking UUID uniqueness"):
                if not check_uuid_uniqueness(data, uuid):
                    print(f"ERROR: UUID {uuid} is not unique in the data")
                    raise ValueError(f"ERROR: UUID {uuid} is not unique in the data")
            
            if all_unique:
                print("All problematic UUIDs are unique. Proceeding with drops...")
                
                # Drop the problematic rollouts
                filtered_data, dropped_count = drop_problematic_rollouts(data, problematic_uuids)
                
                # Save the cleaned data back to the file
                with open(filepath, 'w') as f:
                    for item in filtered_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"Successfully dropped {dropped_count} problematic rollouts from {filename}")
                print(f"Remaining items: {len(filtered_data)}")
            else:
                print(f"Skipping drops for {filename} due to non-unique UUIDs")
        else:
            print("No problematic UUIDs found - no drops needed")

if __name__ == "__main__":
    main()
    # usage: python update_problematic_rollouts.py
