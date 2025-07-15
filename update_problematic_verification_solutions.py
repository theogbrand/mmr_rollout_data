import json
import os
import re

def find_problematic_uuids(data):
    """Find problematic rollout_uuids based on verification solution content"""
    problematic_uuids = []
    
    for item in data:
        # Check if o4_mini_isVerified is False
        if item.get('o4_mini_isVerified') == False:
            rollout_uuid = item.get('rollout_uuid')
            verification_solution = item.get('o4_mini_verification_solution', '')
            
            is_problematic = False
            
            # Check if verification_solution contains required section headers
            has_visual_or_perception = (verification_solution.startswith('[Visual Elements]') or 
                                      verification_solution.startswith('[Perception]'))
            
            # If missing required sections, mark as problematic
            if not has_visual_or_perception:
                is_problematic = True
            else:
                # Additional checks for proper format that parse_first_incorrect_step_from_verification expects
                
                # Check for conclusion tag
                conclusion_pattern = r'<conclusion>(.*?)</conclusion>'
                conclusion_match = re.search(conclusion_pattern, verification_solution, re.DOTALL)
                if not conclusion_match:
                    is_problematic = True
                
                # Check for valid analysis blocks within the expected sections
                if not is_problematic:
                    analysis_blocks = []
                    
                    # Handle Visual Elements/Perception section
                    visual_pattern = r'\[(Visual Elements|Perception)\](.*?)(?=\[Reasoning\]|</conclusion>|$)'
                    for section_match in re.finditer(visual_pattern, verification_solution, re.DOTALL):
                        section_content = section_match.group(2)
                        
                        # Find analysis blocks in this section
                        analysis_pattern = r'<analysis_(\d+)>.*?</analysis_\d+>'
                        for analysis_match in re.finditer(analysis_pattern, section_content, re.DOTALL):
                            step_num = int(analysis_match.group(1))
                            analysis_blocks.append(step_num)
                    
                    # Handle Reasoning section
                    reasoning_pattern = r'\[Reasoning\](.*?)(?=</conclusion>|$)'
                    reasoning_match = re.search(reasoning_pattern, verification_solution, re.DOTALL)
                    if reasoning_match:
                        section_content = reasoning_match.group(1)
                        
                        # Find analysis blocks in reasoning section
                        analysis_pattern = r'<analysis_(\d+)>.*?</analysis_\d+>'
                        for analysis_match in re.finditer(analysis_pattern, section_content, re.DOTALL):
                            step_num = int(analysis_match.group(1))
                            analysis_blocks.append(step_num)
                    
                    # If no analysis blocks found, it's problematic
                    if not analysis_blocks:
                        is_problematic = True
            
            # Add to problematic list if problematic and not already there
            if is_problematic and rollout_uuid not in problematic_uuids:
                problematic_uuids.append(rollout_uuid)
    
    return problematic_uuids

def check_uuid_uniqueness(data, uuid):
    """Check that a UUID appears exactly once in the data"""
    count = sum(1 for item in data if item.get('rollout_uuid') == uuid)
    return count == 1

def update_problematic_items(data, problematic_uuids):
    """Update o4_mini_isVerified from False to None for problematic UUIDs"""
    updated_count = 0
    
    for item in data:
        rollout_uuid = item.get('rollout_uuid')
        if rollout_uuid in problematic_uuids:
            # Verify uniqueness before updating
            if not check_uuid_uniqueness(data, rollout_uuid):
                print(f"ERROR: UUID {rollout_uuid} appears multiple times in data - skipping update")
                raise ValueError(f"ERROR: UUID {rollout_uuid} appears multiple times in data - skipping update")
                continue
            
            if item['o4_mini_isVerified'] != False:
                print(f"ERROR: UUID {rollout_uuid} has o4_mini_isVerified not False - stopping update")
                raise ValueError(f"ERROR: UUID {rollout_uuid} has o4_mini_isVerified not False - stopping update")
                continue
            
            print(f"Updating rollout_uuid: {rollout_uuid}")
            print(f"item: {item}")
            print(f"item['o4_mini_isVerified']: {item['o4_mini_isVerified']}")
            # Update the value
            item['o4_mini_isVerified'] = None
            updated_count += 1
            print(f"Updated rollout_uuid: {rollout_uuid}")
    
    return updated_count

def main():
    # Directory path
    data_dir = "/mnt/fast10/brandon/mmr_rollout_data/final_combined_MC_and_verification_files"
    
    dataset_files = [
        "InfoVQA_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "vqav2_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "CLEVR_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "RAVEN_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "dvqa_final_mc_rollouts_with_all_models_verification_merged.jsonl",
        "AI2D_final_mc_rollouts_with_all_models_verification_merged.jsonl"
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
            for uuid in problematic_uuids:
                if not check_uuid_uniqueness(data, uuid):
                    print(f"ERROR: UUID {uuid} is not unique in the data")
                    raise ValueError(f"ERROR: UUID {uuid} is not unique in the data")
                    all_unique = False
            
            if all_unique:
                print("All problematic UUIDs are unique. Proceeding with updates...")
                
                # Update the problematic items
                updated_count = update_problematic_items(data, problematic_uuids)
                
                # Save the updated data back to the file
                # with open(filepath, 'w') as f:
                #     for item in data:
                #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"Successfully updated {updated_count} items in {filename}")
            else:
                print(f"Skipping updates for {filename} due to non-unique UUIDs")
        else:
            print("No problematic UUIDs found - no updates needed")

if __name__ == "__main__":
    main() 
    # usage: python update_problematic_verification_solutions.py