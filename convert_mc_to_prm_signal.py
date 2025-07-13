import json
import os
from argparse import ArgumentParser
from collections import defaultdict
import re

from constants_and_prompts import PRM_SYSTEM_PROMPT

def transform_image_url_to_s3(image_path):
    """
    Transform image path from local path to S3 URL.
    Replaces '/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/' 
    with 's3://arf-share/arf-ob1-mm-reasoning/training_data_images/'
    """
    # Handle both relative and absolute paths that contain the target pattern
    old_pattern = "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/"
    new_base = "s3://arf-share/arf-ob1-mm-reasoning/training_data_images/"
    
    # Find the pattern in the path and replace it
    if old_pattern in image_path:
        # Split at the pattern and take everything after it
        parts = image_path.split(old_pattern)
        if len(parts) > 1:
            relative_path = parts[-1]  # Take the last part after the pattern
            return new_base + relative_path
    else:
        raise ValueError(f"DEBUG: old_pattern {old_pattern} not found in image_path {image_path}  to replace with {new_base}")
    

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
    visual_elements_match = re.search(r'\[(?:Visual Elements|Perception)\](.*?)\[Reasoning\]', full_rollout_response, re.DOTALL)
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
    
    # Determine the actual first section name from the rollout response
    first_section_name = '[Visual Elements]'  # default
    if re.search(r'\[Perception\]', full_rollout_response):
        first_section_name = '[Perception]'
    
    # Create a mapping of step content (without XML tags) to full XML step and section
    step_to_section_and_xml = {}
    for xml_step in visual_steps:
        # Extract content without XML tags for matching
        content_match = re.search(r'<step_\d+>(.*?)</step_\d+>', xml_step, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
            step_to_section_and_xml[content] = (xml_step, first_section_name)
    
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
        if current_section in ['[Visual Elements]', '[Perception]']:
            visual_elements_step_count += 1
        elif current_section == '[Reasoning]':
            reasoning_step_count += 1

        # Once we find a negative step, all subsequent steps are negative
        if not found_negative and step['score'] <= threshold:
            found_negative = True
            # Record the first incorrect step
            if current_section in ['[Visual Elements]', '[Perception]']:
                first_incorrect_step = ('Visual Elements', visual_elements_step_count - 1)  # Normalize to Visual Elements
            elif current_section == '[Reasoning]':
                first_incorrect_step = ('Reasoning', reasoning_step_count - 1)
        
        conversations.extend([
            {'from': 'human', 'value': step_solution},
            {'from': 'gpt', 'value': '-' if found_negative else '+'},
        ])

        # Early stop after processing the first negative step
        # if first step negative, then conversation only has 1 human-gpt value
        # for socres [0.8, 0.7, 0.6, -0.1, 0.5] and threshold 0.0, step 4 is the final step in the conversation
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
            print(f"DEBUG: {col} has isVerified=True, but MC found incorrect a negative step, for id {target_id}, where full item is {matching_item} and MC filtered item is {mc_filtered_item}. No consensus on error existence.")
            return False
    
    print(f"DEBUG: Both MC score and LLM judges agrees there is an error for id {target_id}")
    print(f"DEBUG: full item {matching_item} and MC filtered item is {mc_filtered_item}. all consensus for MC and LLM judges for INCORRECT sample, checking if index of first incorrect step is the same next.")
    return True


def is_index_of_first_incorrect_step_for_mc_and_llm_judges_consensus(mc_filtered_item, all_items_array):
    """
    Check if MC and all LLM judges agree on which step is the first incorrect one.
    Assumes is_llm_judges_consensus_for_incorrect has already returned True.
    """
    # Validate that first_incorrect_step is in the expected format
    print(f"DEBUG: now checking if index of first incorrect step is the same for MC and LLM judges")
    mc_first_incorrect_step = mc_filtered_item.get('first_incorrect_step')
    print(f"DEBUG: mc_first_incorrect_step: {mc_first_incorrect_step}")
    if not isinstance(mc_first_incorrect_step, tuple) or len(mc_first_incorrect_step) != 2:
        raise TypeError(f"ERROR: mc_first_incorrect_step must be a tuple of length 2, got {type(mc_first_incorrect_step).__name__}: {mc_first_incorrect_step}")
    
    section, step_index = mc_first_incorrect_step
    if section not in ['Visual Elements', 'Reasoning']:
        raise ValueError(f"ERROR: first element of mc_first_incorrect_step must be 'Visual Elements' or 'Reasoning', got: {section}")
    
    if not isinstance(step_index, int):
        raise TypeError(f"ERROR: second element of mc_first_incorrect_step must be an integer, got {type(step_index).__name__}: {step_index}")
    
    # find the full item in all_items_array that has the same id as the mc_filtered_item
    target_id = mc_filtered_item['id']
    print(f"DEBUG: Checking step consensus for id: {target_id} with MC first_incorrect_step: {mc_first_incorrect_step}")
    
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
        # ('gpt_4.1_mini', 'gpt_4.1_mini_verification_solution'),
        # ('gpt_4.1_nano', 'gpt_4.1_nano_verification_solution')
    ]
    
    # Check each verification model's solution
    for model_name, verification_solution_col in verification_models:
        if verification_solution_col not in mc_matching_full_item:
            raise KeyError(f"ERROR: Column {verification_solution_col} not found in item with id {target_id}")
        
        # Parse the verification solution to find the first incorrect step
        verification_solution = mc_matching_full_item[verification_solution_col]
        print(f"DEBUG: model_name: {model_name}, verification_solution: {verification_solution}")
        try:
            verifier_first_incorrect = parse_first_incorrect_step_from_verification(verification_solution)
        except Exception as e:
            print(f"DEBUG: {model_name} failed to parse verification_solution: {e}. No consensus possible.")
            return False
        
        # Compare with MC's first incorrect step
        if verifier_first_incorrect != mc_first_incorrect_step:
            print(f"DEBUG: {model_name} first incorrect step {verifier_first_incorrect} doesn't match MC {mc_first_incorrect_step}. No consensus on step index.")
            return False
        
        print(f"DEBUG: {model_name} agrees with MC on first incorrect step: {mc_first_incorrect_step}")
    
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
    <conclusion>
    Incorrect
    </conclusion>
    
    ---

    **Explanation:** The visual elements analysis is entirely correct. The reasoning analyses are mostly sound up to step 8. However, the final answer choice is flawed. The solution itself acknowledges that the correct outcome would be fish increase and crabs decrease, making "crabs grow in number" incorrect. Given the question asks to pick from the listed options and reasoning shows none match well, the solution should either indicate no effect or "fish grow in number" if that was an option, or otherwise state that no provided answer is correct.

    Since the final provided answer contradicts the correct biological inference and the solution explicitly states this, the solution is incorrect in the final answer selection.

    Therefore, the proper conclusion is "Incorrect."
    """
    if not verification_solution:
        raise ValueError("ERROR: verification_solution is empty")
    
    # Check that the conclusion is "Incorrect" or "incorrect"
    conclusion_pattern = r'<conclusion>(.*?)</conclusion>'
    conclusion_match = re.search(conclusion_pattern, verification_solution, re.DOTALL)
    
    if not conclusion_match:
        raise ValueError("ERROR: No conclusion tag found in verification_solution")
    
    conclusion_text = conclusion_match.group(1).strip()
    if conclusion_text.lower() != "incorrect":
        raise ValueError(f"ERROR: Expected conclusion to be 'Incorrect' or 'incorrect', got: '{conclusion_text}'")
    
    # Find all analysis blocks with their sections
    analysis_blocks = []
    
    # Handle Visual Elements/Perception section
    visual_pattern = r'\[(Visual Elements|Perception)\](.*?)(?=\[Reasoning\]|</conclusion>|$)'
    for section_match in re.finditer(visual_pattern, verification_solution, re.DOTALL):
        section_name = section_match.group(1)
        section_content = section_match.group(2)
        
        # Find analysis blocks in this section
        analysis_pattern = r'<analysis_(\d+)>.*?</analysis_\d+>'
        for analysis_match in re.finditer(analysis_pattern, section_content, re.DOTALL):
            step_num = int(analysis_match.group(1)) # note the step_num starts from 1 here so we need to subtract 1 to get the 0-based index
            analysis_blocks.append((section_name, step_num))
    
    # Handle Reasoning section
    reasoning_pattern = r'\[Reasoning\](.*?)(?=</conclusion>|$)'
    reasoning_match = re.search(reasoning_pattern, verification_solution, re.DOTALL)
    if reasoning_match:
        section_content = reasoning_match.group(1)
        
        # Find analysis blocks in reasoning section
        analysis_pattern = r'<analysis_(\d+)>.*?</analysis_\d+>'
        for analysis_match in re.finditer(analysis_pattern, section_content, re.DOTALL):
            step_num = int(analysis_match.group(1)) # note the step_num starts from 1 here so we need to subtract 1 to get the 0-based index
            analysis_blocks.append(("Reasoning", step_num))
    
    if not analysis_blocks:
        raise ValueError("ERROR: No analysis block found in verification_solution")
    
    # Return the last analysis block (first incorrect step)
    last_section, last_step_num = analysis_blocks[-1]
    # Normalize Perception to Visual Elements for consistency
    if last_section == 'Perception':
        last_section = 'Visual Elements'
    print(f"DEBUG: returning (last_section, last_step_num - 1): {(last_section, last_step_num - 1)}")
    return (last_section, last_step_num - 1)  # Convert to 0-based


def check_all_step_correct_consensus(mc_filtered_item: dict, all_items_array: list[dict], verification_columns: list[str]) -> bool:
    target_id = mc_filtered_item['id']
    
    print(f"DEBUG: Looking for item with id: {target_id}")
    
    # Find the matching item in all_items_array - ensure exactly one match
    matching_items = [item for item in all_items_array if item.get('rollout_uuid') == target_id]  # based on item2conv_prm, id comes from rollout_uuid
    
    if len(matching_items) == 0:
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        raise ValueError(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
    
    matching_item = matching_items[0]
    print(f"DEBUG: Found ONLY ONE matching item for id: {target_id}")
    
    # Check the verification columns
    # verification_columns = ['o4_mini_isVerified', 'gpt_4.1_mini_isVerified', 'gpt_4.1_nano_isVerified']
    
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
    
    print(f"DEBUG: verification columns {verification_columns} are all True for id {target_id}: {all_verified}")
    return all_verified


def is_LLM_judge_consensus_filtering(mc_filtered_item, all_items_array):
    # if mc filtered item is correct; consensus for correct
    if mc_filtered_item['first_incorrect_step'] is None: # TODO: collect IDs here to count and check manually
        print(f"DEBUG: Judging a trace where all MC steps are correct with threshold selected")
        # just need to check if {model_name}_isVerified is True for all models
            # if check is True, then return True
        return check_all_step_correct_consensus(mc_filtered_item, all_items_array, ['o4_mini_isVerified', 'gpt_4.1_mini_isVerified', 'gpt_4.1_nano_isVerified'])

    else: # check if first incorrect step is the same for verification traces; consensus for incorrect
        print(f"DEBUG: Judging a trace where there is an incorrect step in the trace, first check if LLM judges agree there is an incorrect step, then check if the index of the first incorrect step is the same for MC and LLM judges")
        if is_llm_judges_consensus_for_incorrect(mc_filtered_item, all_items_array): # TODO: collect IDs here to count and check manually
            return is_index_of_first_incorrect_step_for_mc_and_llm_judges_consensus(mc_filtered_item, all_items_array)# TODO: collect IDs here to count and check manually
        else:
            return False

def mc_consensus_filtering_v2_algo(raw_none_null_verification_rollout_item: dict, all_items_array: list[dict]) -> dict:
    print(f"DEBUG: Running v2 consensus filtering algo on item: {raw_none_null_verification_rollout_item}")

    # MC threshold and o4-mini agree on all steps correct:
    mc_filtered_item = item2conv_prm(raw_none_null_verification_rollout_item) # outputs ["first_incorrect_step"] = None if all steps are correct, otherwise (section, step_index) of first incorrect step based on MC threshold
    if mc_filtered_item['first_incorrect_step'] is None:
        print(f"DEBUG: Judging a trace where all MC steps are correct with threshold config set to {args.mc_threshold}")
        if check_all_step_correct_consensus(mc_filtered_item, all_items_array, ['o4_mini_isVerified']): # only choose o4_mini
            print(f"DEBUG: MC threshold and o4-mini agree on all steps correct")
            print(f"DEBUG: returning mc_filtered_item with MC and o4-mini agree on all steps correct: {mc_filtered_item}")
            return mc_filtered_item
        else:
            print(f"DEBUG: Returning None because MC and o4-mini do not agree on all steps correct: {mc_filtered_item}")
            print(f"DEBUG: MC threshold and o4-mini disagree on all steps correct")
            # TODO: Implement to take raw_none_null_verification_rollout_item, use o4-mini identified first incorrect step, and output it in the same share_gpt format style as mc_filtered_item, before goes into final TRL filter 
            exit(0)
            return None
    else:
        print(f"DEBUG: Judging a trace where there is an incorrect step in the trace, since by MC score and o4-mini it is not a correct trace.\nWe ignore the first incorrect step identified by MC threshold and only use o4-mini to identify the first incorrect step")
        # identify the first incorrect step based on o4-mini and output it in the same share_gpt format style mc_filtered_item before goes into final TRL filter 
        # TODO: implement this

        return None


# follow TRL expected data format
def final_filter_and_processing_before_training(final_mc_prm_data): # final_mc_prm_data input df columns: (['id', 'image_url', 'conversations', 'first_incorrect_step', 'steps_with_score'])
    """
    Convert from ShareGPT format to TRL format
    - ShareGPT: {'from': 'human', 'value': 'text'}
    - TRL: {'role': 'user', 'content': [{'type': 'text', 'text': 'text', 'index': None}]}
    """
    # Convert conversations from ShareGPT to TRL format
    trl_messages = []
    image_added = False
    
    for msg in final_mc_prm_data['conversations']:
        # Map ShareGPT roles to TRL roles
        role_mapping = {
            'human': 'user',
            'gpt': 'assistant', 
            'system': 'system'
        }
        
        trl_role = role_mapping.get(msg['from'], msg['from'])
        
        # Create content list with text
        content = [{"type": "text", "text": msg['value'], "index": None}]
        
        # Add image to the first human/user message
        if trl_role == 'user' and not image_added:
            content.append({"type": "image", "text": None, "index": 0})
            image_added = True
        
        trl_messages.append({
            "role": trl_role,
            "content": content
        })

    # current_image_url = "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/AI2D/subset_images/78.png" 
    
    
    return {
        "messages": trl_messages,
        "images": [transform_image_url_to_s3(final_mc_prm_data['image_url'])],  # List of image URLs/paths
        "id": final_mc_prm_data['id']
    }


def main():
    # print all configs:
    print(f'{args=}')

    if not os.path.exists(args.data_dir):
        print(f'Dir does not exist: {args.data_dir}')
        exit(0)

    for filename in os.listdir(args.data_dir): # TODO: remove later to run on all files
        if not filename.endswith('.jsonl'):
            continue

        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'debug'), exist_ok=True)

        pairs_save_path = os.path.join(save_dir, 'debug', f'{ds_name}_prm_training_data_mc{args.mc_threshold}.jsonl')
        final_trl_format_save_path = os.path.join(save_dir, 'train', f'{ds_name}_prm_training_data_final_trl_format_mc{args.mc_threshold}.jsonl')
        # pairs_orm_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_orm.jsonl')

        if os.path.exists(pairs_save_path) and not args.overwrite:
            print(f'Debug File already exists: {pairs_save_path} or Train file path already exists: {final_trl_format_save_path}')
            print(f'Skipping file: {filename}')
            continue

        info = defaultdict(int)
        # id2scores = defaultdict(list)
        statistics = defaultdict(list)

        convs_prm = [] # "conversations prm for debugging"
        final_trl_format_items = [] # "final trl format items for training"
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

        # Core filtering logic comes here:
        for item in filtered_items:
            if args.consensus_filtering_algo_version == 'v1':
                print(f'Running v1 consensus filtering algo')
                # First, we apply the MC threshold to the item and get the "first incorrect step" based on the threshold
                mc_filtered_item = item2conv_prm(item)
                # add a function to check if mc_filtered_item has first step incorrect. If so, then we can skip the item.
                if is_LLM_judge_consensus_filtering(mc_filtered_item, filtered_items): # TODO: collect IDs here to count and check manually
                    convs_prm.append(mc_filtered_item)
                    final_filtered_item = final_filter_and_processing_before_training(mc_filtered_item)
                    final_trl_format_items.append(final_filtered_item)
                else:
                    continue # track rows that failed the consensus filtering in another array that is saved for auditing
                # TODO: collect IDs here to count and check manually
            elif args.consensus_filtering_algo_version == 'v2':
                print(f'Running v2 consensus filtering algo')
                mc_consensus_filtered_v2_item = mc_consensus_filtering_v2_algo(item, filtered_items) # expected output schema: (['id', 'image_url', 'conversations', 'first_incorrect_step', 'steps_with_score'])

                if mc_consensus_filtered_v2_item is not None:
                    convs_prm.append(mc_consensus_filtered_v2_item)

                    final_filtered_item = final_filter_and_processing_before_training(mc_consensus_filtered_v2_item)

                    final_trl_format_items.append(final_filtered_item)
            else:
                raise ValueError(f"ERROR: Invalid consensus filtering algo version: {args.consensus_filtering_algo_version}")
 
            statistics['num_turns'].append(len(convs_prm[-1]['conversations']))

        print(f'[{filename}]')
        for k, v in info.items():
            print(k, v)
        for k, v in statistics.items():
            print(f'{k}: max={max(v)}, min={min(v)}, mean={sum(v) / len(v)}, total={sum(v)}')
        print()

        save_outputs(convs_prm, pairs_save_path)
        save_outputs(final_trl_format_items, final_trl_format_save_path)
        # if args.include_orm_data:
            # save_outputs(convs_orm, pairs_orm_save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/fast10/brandon/mmr_rollout_data/final_combined_MC_and_verification_files')
    parser.add_argument('--save-dir', type=str, default='/mnt/fast10/brandon/mmr_rollout_data/prm_training_data')
    parser.add_argument('--mc-threshold', type=float, default=0.8) # TODO: try 0.5 and 0.8; and maybe include/exclude nano. Point is to find more "-" points where LLM Judge can agree on it being an error. (0.8 comes from GenPRM recommendation for math reasoning)
    parser.add_argument('--early-stop', action='store_true', default=True)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--consensus-filtering-algo-version', type=str, default='v2') 
    # parser.add_argument('--include-orm-data', action='store_true', default=False)
    args = parser.parse_args()

    main()