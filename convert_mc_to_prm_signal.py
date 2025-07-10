import json
import os
from argparse import ArgumentParser
from collections import defaultdict
import re

from constants_and_prompts import PRM_SYSTEM_PROMPT

def load_outputs(results_file):
    with open(results_file) as file:
        lines = file.readlines()
    items = [json.loads(line) for line in lines]
    return items

def save_outputs(outputs, results_file):
    outputs = sorted(outputs, key=lambda x:str(x['id']))

    with open(results_file, 'w') as file:
        for output in outputs:
            file.write(json.dumps(output) + '\n')

    print(f'Results ({len(outputs)=}) saved to {results_file}')


def item2conv_prm(item):
    id = item['rollout_uuid']
    image = item['rollout_image_path']
    question = item['rollout_question']
    full_rollout_response = item['rollout_response']
    steps_with_score = item['rollout_steps_with_score']

    threshold = args.mc_threshold
    conversations = [{'from': 'system', 'value': PRM_SYSTEM_PROMPT}]
    found_negative = False
    first_incorrect_step = None
    
    # Find section boundaries
    visual_elements_match = re.search(r'\[Visual Elements\](.*?)\[Reasoning\]', full_rollout_response, re.DOTALL)
    reasoning_match = re.search(r'\[Reasoning\](.*?)(?:<correct_answer>|$)', full_rollout_response, re.DOTALL)
    
    # Extract steps from each section with XML tags preserved
    visual_steps = []
    reasoning_steps = []
    
    if visual_elements_match:
        visual_content = visual_elements_match.group(1)
        visual_step_matches = re.findall(r'(<step_\d+>.*?</step_\d+>)', visual_content, re.DOTALL)
        visual_steps = [step.strip() for step in visual_step_matches]
    
    if reasoning_match:
        reasoning_content = reasoning_match.group(1)
        reasoning_step_matches = re.findall(r'(<step_\d+>.*?</step_\d+>)', reasoning_content, re.DOTALL)
        reasoning_steps = [step.strip() for step in reasoning_step_matches]
    
    # Create a mapping of step content (without XML tags) to full XML step and section
    step_to_section_and_xml = {}
    for xml_step in visual_steps:
        # Extract content without XML tags for matching
        content_match = re.search(r'<step_\d+>(.*?)</step_\d+>', xml_step, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
            step_to_section_and_xml[content] = (xml_step, '[Visual Elements]')
    
    for xml_step in reasoning_steps:
        # Extract content without XML tags for matching
        content_match = re.search(r'<step_\d+>(.*?)</step_\d+>', xml_step, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
            step_to_section_and_xml[content] = (xml_step, '[Reasoning]')
    
    # Process each scored step
    current_section = None
    visual_elements_step_count = 0
    reasoning_step_count = 0
    
    for step_idx, step in enumerate(steps_with_score):
        step_solution = step['step'].strip()
        
        if step_idx == 0:
            # First step includes the question and solution process header with Visual Elements section
            if step_solution in step_to_section_and_xml:
                xml_step, section = step_to_section_and_xml[step_solution]
                step_solution = f'### Question:\n{question}\n\n### Solution Process:\n[Visual Elements]\n{xml_step}'
                current_section = section
            else:
                raise ValueError(f"Step solution not found in step_to_section_and_xml: {step_solution}")
        else:
            # Check if this step has XML tags and section info
            if step_solution in step_to_section_and_xml:
                xml_step, section = step_to_section_and_xml[step_solution]
                if section != current_section:
                    # Prepend section header to the XML step
                    step_solution = f'{section}\n{xml_step}'
                    current_section = section
                else:
                    # Use XML step without section header
                    step_solution = xml_step
        
        # Update step counters based on current section
        if current_section == '[Visual Elements]':
            visual_elements_step_count += 1
        elif current_section == '[Reasoning]':
            reasoning_step_count += 1

        # Once we find a negative step, all subsequent steps are negative
        if not found_negative and step['score'] <= threshold:
            found_negative = True
            # Record the first incorrect step
            if current_section == '[Visual Elements]':
                first_incorrect_step = ('Visual Elements', visual_elements_step_count - 1)
            elif current_section == '[Reasoning]':
                first_incorrect_step = ('Reasoning', reasoning_step_count - 1)
        
        conversations.extend([
            {'from': 'human', 'value': step_solution},
            {'from': 'gpt', 'value': '-' if found_negative else '+'},
        ])

        if args.early_stop and step['score'] <= threshold:
            break

    return {
        'id': id,
        'image_url': image, # name follows process_vision_info qwen function requirement: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L321
        'conversations': conversations,
        'first_incorrect_step': first_incorrect_step, # None if all steps are correct, otherwise (section, step_index)
        'steps_with_score': steps_with_score,
    }


# Takes last step's MC score as the reference for training signal
# def item2conv_orm(item):
#     id = item['id']
#     image = item['image_path']
#     question = item['question']
#     steps_with_score = item['steps_with_score']

#     if 'response' in item:
#         response = item['response']
#     else:
#         response = '\n\n'.join([step['step'] for step in steps_with_score]).strip()

#     query = f'### Question:\n{question}\n\n### Solution Process:\n{response}'
#     last_step_score = steps_with_score[-1]['score']

#     threshold = args.mc_threshold
#     conversations = [
#         {'from': 'system', 'value': PRM_SYSTEM_PROMPT},
#         {'from': 'human', 'value': query},
#         {'from': 'gpt', 'value': '+' if last_step_score > threshold else '-'},
#     ]

#     return {
#         'id': id,
#         'image_path': image,
#         'conversations': conversations,
#     }


def is_llm_judges_consensus_for_incorrect(mc_filtered_item, all_items_array):
    """
    Check if all LLM judges agree that there is an incorrect step in the solution.
    Returns True if all verifiers have isVerified=False, False otherwise.
    """
    target_id = mc_filtered_item['id']
    
    # Find the matching item in all_items_array - ensure exactly one match
    matching_items = [item for item in all_items_array if item.get('rollout_uuid') == target_id]
    
    if len(matching_items) == 0:
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        raise ValueError(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
    
    matching_item = matching_items[0]
    
    # Check all verification columns
    verification_columns = ['o4_mini_isVerified', 'gpt_4.1_mini_isVerified', 'gpt_4.1_nano_isVerified']
    
    for col in verification_columns:
        if col not in matching_item:
            raise KeyError(f"ERROR: Column {col} not found in item with id {target_id}")
        
        is_verified = matching_item[col]
        if not isinstance(is_verified, bool):
            raise TypeError(f"ERROR: Column {col} is not boolean (got {type(is_verified).__name__}: {is_verified}) for item with id {target_id}")
        
        # If any verifier thinks all steps are correct, consensus fails
        if is_verified:
            # print(f"DEBUG: {col} has isVerified=True, but MC found incorrect a negative step, for id {target_id}, where full item is {matching_item} and MC filtered item is {mc_filtered_item}. No consensus on error existence.")
            return False
    
    print(f"DEBUG: Both MC score and LLM judges agrees there is an error for id {target_id}")
    print(f"DEBUG: full item {matching_item} and MC filtered item is {mc_filtered_item}. all consensus for MC and LLM judges for INCORRECT sample, checking if index of first incorrect step is the same next.")
    exit() # TODOL Continue from here
    return True


def is_index_of_first_incorrect_step_for_mc_and_llm_judges_consensus(mc_filtered_item, all_items_array):
    """
    Check if MC and all LLM judges agree on which step is the first incorrect one.
    Assumes is_llm_judges_consensus_for_incorrect has already returned True.
    """
    # Validate that first_incorrect_step is in the expected format
    first_incorrect_step = mc_filtered_item.get('first_incorrect_step')
    if not isinstance(first_incorrect_step, tuple) or len(first_incorrect_step) != 2:
        raise TypeError(f"ERROR: first_incorrect_step must be a tuple of length 2, got {type(first_incorrect_step).__name__}: {first_incorrect_step}")
    
    section, step_index = first_incorrect_step
    if section not in ['Visual Elements', 'Reasoning']:
        raise ValueError(f"ERROR: first element of first_incorrect_step must be 'Visual Elements' or 'Reasoning', got: {section}")
    
    if not isinstance(step_index, int):
        raise TypeError(f"ERROR: second element of first_incorrect_step must be an integer, got {type(step_index).__name__}: {step_index}")
    
    # find the full item in all_items_array that has the same id as the mc_filtered_item
    target_id = mc_filtered_item['id']
    print(f"DEBUG: Checking step consensus for id: {target_id} with MC first_incorrect_step: {first_incorrect_step}")
    
    # Find the matching item in all_items_array - ensure exactly one match
    matching_items = [item for item in all_items_array if item.get('rollout_uuid') == target_id]
    
    if len(matching_items) == 0:
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        raise ValueError(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
    
    mc_matching_full_item = matching_items[0]
    
    # Define the verification model pairs
    verification_models = [
        ('o4_mini', 'o4_mini_verification_solution'),
        ('gpt_4.1_mini', 'gpt_4.1_mini_verification_solution'),
        ('gpt_4.1_nano', 'gpt_4.1_nano_verification_solution')
    ]
    
    # Check each verification model's solution
    for model_name, verification_solution_col in verification_models:
        if verification_solution_col not in mc_matching_full_item:
            raise KeyError(f"ERROR: Column {verification_solution_col} not found in item with id {target_id}")
        
        # Parse the verification solution to find the first incorrect step
        verification_solution = mc_matching_full_item[verification_solution_col]
        try:
            verifier_first_incorrect = parse_first_incorrect_step_from_verification(verification_solution)
        except Exception as e:
            print(f"DEBUG: {model_name} failed to parse verification_solution: {e}. No consensus possible.")
            return False
        
        # Compare with MC's first incorrect step
        if verifier_first_incorrect != first_incorrect_step:
            print(f"DEBUG: {model_name} first incorrect step {verifier_first_incorrect} doesn't match MC {first_incorrect_step}. No consensus on step index.")
            return False
        
        print(f"DEBUG: {model_name} agrees with MC on first incorrect step: {first_incorrect_step}")
    
    # All models agree with MC on the first incorrect step
    print(f"DEBUG: All verification models agree with MC on first incorrect step for id {target_id}")
    return True


def parse_first_incorrect_step_from_verification(verification_solution):
    """
    Parse the verification solution to find the first incorrect step.
    Since the verification process stops after finding the first incorrect step,
    we return the section and step index of the LAST analysis block found.
    
    Returns a tuple (section, step_index) where section is 'Visual Elements' or 'Reasoning'
    and step_index is 0-based.
    
    Example format:
    [Visual Elements]
    <analysis_1>
    Step 1 correctly lists...
    </analysis_1>
    <analysis_2>
    Step 2 correctly notes...
    </analysis_2>
    <analysis_3>
    Step 3 is incorrect...  # This is the last analysis block - the first incorrect step
    </analysis_3>
    """
    if not verification_solution:
        raise ValueError("ERROR: verification_solution is empty")
    
    # Track current section and last analysis block found
    current_section = None
    last_analysis_section = None
    last_analysis_step_num = None
    
    # Split into lines for processing
    lines = verification_solution.split('\n')
    
    # Regular expressions for parsing
    section_pattern = re.compile(r'^\[(Visual Elements|Reasoning)\]$')
    analysis_pattern = re.compile(r'^<analysis_(\d+)>$')
    analysis_end_pattern = re.compile(r'^</analysis_\d+>$')
    
    in_analysis = False
    current_step_num = None
    
    for line in lines:
        line = line.strip()
        
        # Check for section headers
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue
        
        # Check for analysis start
        analysis_match = analysis_pattern.match(line)
        if analysis_match:
            in_analysis = True
            current_step_num = int(analysis_match.group(1))
            continue
        
        # Check for analysis end
        if analysis_end_pattern.match(line):
            if in_analysis and current_step_num is not None:
                # Update the last analysis block found
                last_analysis_section = current_section
                last_analysis_step_num = current_step_num
            
            in_analysis = False
            current_step_num = None
            continue
    
    # Return the last analysis block found (which is the first incorrect step)
    if last_analysis_section is not None and last_analysis_step_num is not None:
        # Convert to 0-based index
        return (last_analysis_section, last_analysis_step_num - 1)
    
    # If we reach here, no analysis block was found
    raise ValueError("ERROR: No analysis block found in verification_solution")


def check_all_step_correct_consensus(mc_filtered_item, all_items_array):
    target_id = mc_filtered_item['id']
    
    print(f"DEBUG: Looking for item with id: {target_id}")
    
    # Find the matching item in all_items_array - ensure exactly one match
    matching_items = [item for item in all_items_array if item.get('rollout_uuid') == target_id]  # based on item2conv_prm, id comes from rollout_uuid
    
    if len(matching_items) == 0:
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        raise ValueError(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
    
    matching_item = matching_items[0]
    print(f"DEBUG: Found matching item for id: {target_id}")
    
    # Check the verification columns
    verification_columns = ['o4_mini_isVerified', 'gpt_4.1_mini_isVerified', 'gpt_4.1_nano_isVerified']
    
    verification_status = {}
    for col in verification_columns:
        if col not in matching_item:
            raise KeyError(f"ERROR: Column {col} not found in item with id {target_id}")
        
        value = matching_item[col]
        if not isinstance(value, bool):
            raise TypeError(f"ERROR: Column {col} is not boolean (got {type(value).__name__}: {value}) for item with id {target_id}")
        
        verification_status[col] = value
    
    print(f"DEBUG: Verification status for id {target_id}: {verification_status}")
    
    # Check if all verification columns are True
    all_verified = all(verification_status[col] for col in verification_columns)
    
    print(f"DEBUG: All verification columns True for id {target_id}: {all_verified}")
    return all_verified


def is_LLM_judge_consensus_filtering(mc_filtered_item, all_items_array):
    # if mc filtered item is correct; consensus for correct
    if mc_filtered_item['first_incorrect_step'] is None:
        print(f"DEBUG: Judging a trace where all MC steps are correct with threshold selected")
        # just need to check if {model_name}_isVerified is True for all models
            # if check is True, then return True
        return check_all_step_correct_consensus(mc_filtered_item, all_items_array)

    else: # check if first incorrect step is the same for verification traces; consensus for incorrect
        print(f"DEBUG: Judging a trace where there is an incorrect step in the trace, first check if LLM judges agree there is an incorrect step, then check if the index of the first incorrect step is the same for MC and LLM judges")
        if is_llm_judges_consensus_for_incorrect(mc_filtered_item, all_items_array):
            return is_index_of_first_incorrect_step_for_mc_and_llm_judges_consensus(mc_filtered_item, all_items_array)
        else:
            return False


# follow TRL expected data format
def final_filter_and_processing_before_training(final_mc_prm_data):
    return


def main():
    # print all configs:
    print(f'{args=}')

    if not os.path.exists(args.data_dir):
        print(f'Dir does not exist: {args.data_dir}')
        exit(0)

    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.jsonl'):
            continue

        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)

        pairs_save_path = os.path.join(save_dir, 'train', f'{ds_name}_prm_training_data.jsonl')
        # pairs_orm_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_orm.jsonl')

        if os.path.exists(pairs_save_path) and not args.overwrite:
            continue

        info = defaultdict(int)
        # id2scores = defaultdict(list)
        statistics = defaultdict(list)

        convs_prm = []
        # convs_orm = []
        items = load_outputs(os.path.join(args.data_dir, filename))

        # Filter out items with None values in verification columns first
        verification_columns = ['o4_mini_isVerified', 'gpt_4.1_mini_isVerified', 'gpt_4.1_nano_isVerified']
        filtered_items = []
        for item in items:
            if any(col not in item or item[col] is None for col in verification_columns):
                print(f"DEBUG: Skipping item because it has None values in verification columns")
                continue
            filtered_items.append(item)

        # for item in items:
        #     image = item['image_path']
        #     question = item['question']
        #     steps_with_score = item['steps_with_score']

        #     score = steps_with_score[-1]['score']
        #     id2scores[(str(image), question)].append(score)

        for item in filtered_items:
            mc_filtered_item = item2conv_prm(item)
            if is_LLM_judge_consensus_filtering(mc_filtered_item, filtered_items):
                convs_prm.append(mc_filtered_item)
                # final_filtered_item = final_filter_and_processing_before_training(mc_filtered_item)
                # convs_prm.append(final_filtered_item)
                # print(convs_prm)
                # exit()
            else:
                continue # track rows that failed the consensus filtering in another array that is saved for auditing
 
            statistics['num_turns'].append(len(convs_prm[-1]['conversations']))

        print(f'[{filename}]')
        for k, v in info.items():
            print(k, v)
        for k, v in statistics.items():
            print(f'{k}: max={max(v)}, min={min(v)}, mean={sum(v) / len(v)}, total={sum(v)}')
        print()

        save_outputs(convs_prm, pairs_save_path)
        # if args.include_orm_data:
            # save_outputs(convs_orm, pairs_orm_save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/fast10/brandon/mmr_rollout_data/final_combined_MC_and_verification_files')
    parser.add_argument('--save-dir', type=str, default='/mnt/fast10/brandon/mmr_rollout_data/prm_training_data')
    parser.add_argument('--mc-threshold', type=float, default=0.0)
    parser.add_argument('--early-stop', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    # parser.add_argument('--include-orm-data', action='store_true', default=False)
    args = parser.parse_args()

    main()