import json
import os
import logging
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict
import re
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

from constants_and_prompts import PRM_SYSTEM_PROMPT

# Set up logging
def setup_logging():
    log_dir = "consensus_filtering_to_prm_signal_conversion_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"conversion_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


def convert_image_to_base64_string(image_path):
    """Convert local image to base64 JPEG string."""
    try:
        with Image.open(image_path) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error converting image {image_path} to base64: {e}")
        raise ValueError(f"Error converting image {image_path} to base64: {e}")


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
        logger.error(f"DEBUG: old_pattern {old_pattern} not found in image_path {image_path}  to replace with {new_base}")
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
            file.write(json.dumps(output, ensure_ascii=False) + '\n')

    logger.info(f'Results ({len(outputs)=}) saved to {results_file}')


def item2conv_prm(item):
    id = item['rollout_uuid']
    image = item['rollout_image_path']
    question_match = re.search(r'<question>(.*?)</question>', item['rollout_question'], re.DOTALL)
    question = question_match.group(1).strip() if question_match else None
    if question is None:
        logger.error(f"ERROR: No question found in rollout_question: {item['rollout_question']}")
        raise ValueError(f"ERROR: No question found in rollout_question: {item['rollout_question']}")
    full_rollout_response = item['rollout_response'] # this function puts the full rollout response into a conversations format
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
                logger.error(f"Step solution not found in step_to_section_and_xml: {step_solution}")
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
        logger.error(f"ERROR: Could not find item with id {target_id} in all_items_array")
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        logger.error(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
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
            logger.debug(f"{col} has isVerified=True, but MC found incorrect a negative step, for id {target_id}, where full item is {matching_item} and MC filtered item is {mc_filtered_item}. No consensus on error existence.")
            return False
    
    logger.debug(f"Both MC score and LLM judges agrees there is an error for id {target_id}")
    logger.debug(f"full item {matching_item} and MC filtered item is {mc_filtered_item}. all consensus for MC and LLM judges for INCORRECT sample, checking if index of first incorrect step is the same next.")
    return True


def is_index_of_first_incorrect_step_for_mc_and_llm_judges_consensus(mc_filtered_item, all_items_array):
    """
    Check if MC and all LLM judges agree on which step is the first incorrect one.
    Assumes is_llm_judges_consensus_for_incorrect has already returned True.
    """
    # Validate that first_incorrect_step is in the expected format
    logger.debug(f"now checking if index of first incorrect step is the same for MC and LLM judges")
    mc_first_incorrect_step = mc_filtered_item.get('first_incorrect_step')
    logger.debug(f"mc_first_incorrect_step: {mc_first_incorrect_step}")
    if not isinstance(mc_first_incorrect_step, tuple) or len(mc_first_incorrect_step) != 2:
        raise TypeError(f"ERROR: mc_first_incorrect_step must be a tuple of length 2, got {type(mc_first_incorrect_step).__name__}: {mc_first_incorrect_step}")
    
    section, step_index = mc_first_incorrect_step
    if section not in ['Visual Elements', 'Reasoning']:
        logger.error(f"ERROR: first element of mc_first_incorrect_step must be 'Visual Elements' or 'Reasoning', got: {section}")
        raise ValueError(f"ERROR: first element of mc_first_incorrect_step must be 'Visual Elements' or 'Reasoning', got: {section}")
    
    if not isinstance(step_index, int):
        raise TypeError(f"ERROR: second element of mc_first_incorrect_step must be an integer, got {type(step_index).__name__}: {step_index}")
    
    # find the full item in all_items_array that has the same id as the mc_filtered_item
    target_id = mc_filtered_item['id']
    logger.debug(f"Checking step consensus for id: {target_id} with MC first_incorrect_step: {mc_first_incorrect_step}")
    
    # Find the matching item in all_items_array, so we can get the verification solution, it was filtered out by the MC threshold filtering condition - ensure exactly one match
    matching_items = [item for item in all_items_array if item.get('rollout_uuid') == target_id]
    
    if len(matching_items) == 0:
        logger.error(f"ERROR: Could not find item with id {target_id} in all_items_array")
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        logger.error(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
        raise ValueError(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
    
    mc_matching_full_item = matching_items[0]
    
    # Adjust this based on which verifier models we want to use for consensus. Can be from 1 to 3 models.
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
        logger.debug(f"model_name: {model_name}, verification_solution: {verification_solution}")
        try:
            verifier_first_incorrect = parse_first_incorrect_step_from_verification(verification_solution)
        except Exception as e:
            logger.debug(f"{model_name} failed to parse verification_solution: {e}. No consensus possible.")
            return False
        
        # Compare with MC's first incorrect step
        if verifier_first_incorrect != mc_first_incorrect_step:
            logger.debug(f"{model_name} first incorrect step {verifier_first_incorrect} doesn't match MC {mc_first_incorrect_step}. No consensus on step index.")
            return False
        
        logger.debug(f"{model_name} agrees with MC on first incorrect step: {mc_first_incorrect_step}")
    
    # All models agree with MC on the first incorrect step
    logger.debug(f"All verification models agree with MC on first incorrect step for id {target_id}")
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
        logger.error("ERROR: verification_solution is empty")
        raise ValueError("ERROR: verification_solution is empty")
    
    # Check that the conclusion is "Incorrect" or "incorrect"
    conclusion_pattern = r'<conclusion>(.*?)</conclusion>'
    conclusion_match = re.search(conclusion_pattern, verification_solution, re.DOTALL)
    
    if not conclusion_match:
        logger.error("ERROR: No conclusion tag found in verification_solution")
        raise ValueError("ERROR: No conclusion tag found in verification_solution")
    
    conclusion_text = conclusion_match.group(1).strip()
    if conclusion_text.lower() != "incorrect":
        logger.error(f"ERROR: Expected conclusion to be 'Incorrect' or 'incorrect', got: '{conclusion_text}'")
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
    
    if not analysis_blocks: # happens when verification solution does not have the [Visual Elements] or [Reasoning] section but has a conclusion. we manually change their o4_mini_isVerified to None manually using edit_final_combined_MC_and_verification_files_manually.ipynb for now
        logger.error("ERROR: No analysis block found in verification_solution")
        raise ValueError("ERROR: No analysis block found in verification_solution")
    
    # Return the last analysis block (first incorrect step)
    last_section, last_step_num = analysis_blocks[-1]
    # Normalize Perception to Visual Elements for consistency
    if last_section == 'Perception':
        last_section = 'Visual Elements'
    logger.debug(f"returning (last_section, last_step_num - 1): {(last_section, last_step_num - 1)}")
    return (last_section, last_step_num - 1)  # Convert to 0-based

def parse_first_correct_step_from_verification(verification_solution):
    """
    Parse the verification solution to find the first correct step.
    Since the verification process stops after finding the first correct step,
    we return the section and step index of the FIRST analysis block found.
    
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
    Correct
    </conclusion>
    
    ---

    **Explanation:** The visual elements analysis is entirely correct. The reasoning analyses are mostly sound up to step 8. However, the final answer choice is flawed. The solution itself acknowledges that the correct outcome would be fish increase and crabs decrease, making "crabs grow in number" incorrect. Given the question asks to pick from the listed options and reasoning shows none match well, the solution should either indicate no effect or "fish grow in number" if that was an option, or otherwise state that no provided answer is correct.

    Since the final provided answer contradicts the correct biological inference and the solution explicitly states this, the solution is incorrect in the final answer selection. 

    Therefore, the proper conclusion is "Correct."
    """
    if not verification_solution:
        logger.error("ERROR: verification_solution is empty")
        raise ValueError("ERROR: verification_solution is empty")
    
    # Check that the conclusion is "Incorrect" or "incorrect"
    conclusion_pattern = r'<conclusion>(.*?)</conclusion>'
    conclusion_match = re.search(conclusion_pattern, verification_solution, re.DOTALL)
    
    if not conclusion_match:
        logger.error("ERROR: No conclusion tag found in verification_solution")
        raise ValueError("ERROR: No conclusion tag found in verification_solution")
    
    conclusion_text = conclusion_match.group(1).strip()
    if conclusion_text.lower() != "correct":
        logger.error(f"ERROR: Expected conclusion to be 'Correct' or 'correct', got: '{conclusion_text}'")
        raise ValueError(f"ERROR: Expected conclusion to be 'Correct' or 'correct', got: '{conclusion_text}'")
    
    # # Find all analysis blocks with their sections
    # analysis_blocks = []
    
    # # Handle Visual Elements/Perception section
    # visual_pattern = r'\[(Visual Elements|Perception)\](.*?)(?=\[Reasoning\]|</conclusion>|$)'
    # for section_match in re.finditer(visual_pattern, verification_solution, re.DOTALL):
    #     section_name = section_match.group(1)
    #     section_content = section_match.group(2)
        
    #     # Find analysis blocks in this section
    #     analysis_pattern = r'<analysis_(\d+)>.*?</analysis_\d+>'
    #     for analysis_match in re.finditer(analysis_pattern, section_content, re.DOTALL):
    #         step_num = int(analysis_match.group(1)) # note the step_num starts from 1 here so we need to subtract 1 to get the 0-based index
    #         analysis_blocks.append((section_name, step_num))
    
    # # Handle Reasoning section
    # reasoning_pattern = r'\[Reasoning\](.*?)(?=</conclusion>|$)'
    # reasoning_match = re.search(reasoning_pattern, verification_solution, re.DOTALL)
    # if reasoning_match:
    #     section_content = reasoning_match.group(1)
        
    #     # Find analysis blocks in reasoning section
    #     analysis_pattern = r'<analysis_(\d+)>.*?</analysis_\d+>'
    #     for analysis_match in re.finditer(analysis_pattern, section_content, re.DOTALL):
    #         step_num = int(analysis_match.group(1)) # note the step_num starts from 1 here so we need to subtract 1 to get the 0-based index
    #         analysis_blocks.append(("Reasoning", step_num))
    
    # if not analysis_blocks:
    #     raise ValueError("ERROR: No analysis block found in verification_solution")
    
    # # Return the last analysis block (last correct step)
    # last_section, last_step_num = analysis_blocks[-1]
    # # Normalize Perception to Visual Elements for consistency
    # if last_section == 'Perception':
    #     last_section = 'Visual Elements'
    # logger.debug(f"returning (last_section, last_step_num - 1): {(last_section, last_step_num - 1)}")
    return (None, None)  # Convert to 0-based


def check_all_step_correct_consensus(mc_filtered_item: dict, all_items_array: list[dict], verification_columns: list[str]) -> bool:
    target_id = mc_filtered_item['id']
    
    logger.debug(f"Looking for item with id: {target_id}")
    
    # Find the matching item in all_items_array - ensure exactly one match
    matching_items = [item for item in all_items_array if item.get('rollout_uuid') == target_id]  # based on item2conv_prm, id comes from rollout_uuid
    
    if len(matching_items) == 0:
        logger.error(f"ERROR: Could not find item with id {target_id} in all_items_array")
        raise ValueError(f"ERROR: Could not find item with id {target_id} in all_items_array")
    elif len(matching_items) > 1:
        logger.error(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
        raise ValueError(f"ERROR: Found {len(matching_items)} items with id {target_id} in all_items_array, expected exactly 1")
    
    matching_item = matching_items[0]
    logger.debug(f"Found ONLY ONE matching item for id: {target_id}")
    
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
    
    logger.debug(f"Verification status for id {target_id}: {verification_status}")
    
    # Check if all verification columns are True
    all_verified = all(verification_status[col] for col in verification_columns)
    
    logger.debug(f"verification columns {verification_columns} are all True for id {target_id}: {all_verified}")
    return all_verified


def is_LLM_judge_consensus_filtering(mc_filtered_item, all_items_array):
    # if mc filtered item is correct; consensus for correct
    if mc_filtered_item['first_incorrect_step'] is None: # TODO: collect IDs here to count and check manually
        logger.debug(f"Judging a trace where all MC steps are correct with threshold selected")
        # just need to check if {model_name}_isVerified is True for all models
            # if check is True, then return True
        return check_all_step_correct_consensus(mc_filtered_item, all_items_array, ['o4_mini_isVerified', 'gpt_4.1_mini_isVerified', 'gpt_4.1_nano_isVerified'])

    else: # check if first incorrect step is the same for verification traces; consensus for incorrect
        logger.debug(f"Judging a trace where there is an incorrect step in the trace, first check if LLM judges agree there is an incorrect step, then check if the index of the first incorrect step is the same for MC and LLM judges")
        if is_llm_judges_consensus_for_incorrect(mc_filtered_item, all_items_array): # TODO: collect IDs here to count and check manually
            return is_index_of_first_incorrect_step_for_mc_and_llm_judges_consensus(mc_filtered_item, all_items_array)# TODO: collect IDs here to count and check manually
        else:
            return False


def raw_item_to_model_identified_first_incorrect_step(raw_not_null_verification_rollout_item: dict, model_to_identify_first_incorrect_step: str, consensus_filtering_algo_label: str) -> dict:
    logger.debug(f"Converting raw item to model identified first incorrect step")

    id = raw_not_null_verification_rollout_item['rollout_uuid']
    image = raw_not_null_verification_rollout_item['rollout_image_path']
    question_match = re.search(r'<question>(.*?)</question>', raw_not_null_verification_rollout_item['rollout_question'], re.DOTALL)
    question = question_match.group(1).strip() if question_match else None
    if question is None:
        logger.error(f"ERROR: No question found in rollout_question: {raw_not_null_verification_rollout_item['rollout_question']}")
        raise ValueError(f"ERROR: No question found in rollout_question: {raw_not_null_verification_rollout_item['rollout_question']}")
    full_rollout_response = raw_not_null_verification_rollout_item['rollout_response'] # we fetch this so we can label it based on the "first incorrect step" identified by the model_to_identify_first_incorrect_step
    steps_with_score = raw_not_null_verification_rollout_item['rollout_steps_with_score']
    
    verifier_identified_first_incorrect_step = tuple()
    if model_to_identify_first_incorrect_step == 'o4_mini':
        verification_solution = raw_not_null_verification_rollout_item['o4_mini_verification_solution']
        verifier_identified_first_incorrect_step = parse_first_incorrect_step_from_verification(verification_solution) # returns (first_incorrect_section_name, first_incorrect_step_num)  # 0-based
        logger.debug(f"o4-mini identified first incorrect step: {verifier_identified_first_incorrect_step}")
    else:
        logger.error(f"ERROR: Model {model_to_identify_first_incorrect_step} not supported")
        raise ValueError(f"ERROR: Model {model_to_identify_first_incorrect_step} not supported")

    # conversations = [{'from': 'system', 'value': PRM_SYSTEM_PROMPT}]
    # found_negative = False
    # first_incorrect_step = None

    # Now find the first incorrect step from the model verification trace
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
    

    conversations = [{'from': 'system', 'value': PRM_SYSTEM_PROMPT}]
    found_negative = False

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
                logger.error(f"Step solution not found in step_to_section_and_xml: {step_solution}")
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
        # Need to normalize section names for comparison (remove brackets)
        current_section_normalized = current_section.replace('[', '').replace(']', '') if current_section else ''
        if not found_negative and ((current_section_normalized in ['Visual Elements', 'Perception'] and current_section_normalized == verifier_identified_first_incorrect_step[0] and (visual_elements_step_count - 1) == verifier_identified_first_incorrect_step[1]) or (current_section_normalized == 'Reasoning' and current_section_normalized == verifier_identified_first_incorrect_step[0] and (reasoning_step_count - 1) == verifier_identified_first_incorrect_step[1])): # because step_idx starts from 0
            found_negative = True

        conversations.extend([
            {'from': 'human', 'value': step_solution},
            {'from': 'gpt', 'value': '-' if found_negative else '+'},
        ])

        # Early stop after processing the first negative step
        # if first step negative, then conversation only has 1 human-gpt value
        # for socres [0.8, 0.7, 0.6, -0.1, 0.5] and threshold 0.0, step 4 is the final step in the conversation
        if args.early_stop and found_negative:
            break

    return {
        'id': id,
        'image_url': image, # name follows process_vision_info qwen function requirement: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L321
        'conversations': conversations,
        'first_incorrect_step': verifier_identified_first_incorrect_step, # None if all steps are correct, otherwise (section, step_index)
        'steps_with_score': steps_with_score,
        'consensus_filtering_algo_label': consensus_filtering_algo_label,
        'verifier_identified_first_incorrect_step_solution': verification_solution,
    } # return format: # final_mc_prm_data input df columns: (['id', 'image_url', 'conversations', 'first_incorrect_step', 'steps_with_score', "consensus_filtering_algo_label" -> "o4-mini_incorrect_and_MC_agrees_and_disagrees", "o4-mini_correct_and_MC_agrees", "o4-mini_correct_and_MC_disagrees"], "verifier_identified_first_incorrect_step_solution")


# TODO: this is tech debt, temp function until decide what to do with these labels
def raw_item_to_uniform_output_format(raw_not_null_verification_rollout_item: dict, model_to_identify_first_incorrect_step: str, consensus_filtering_algo_label: str) -> dict:
    logger.debug(f"Converting raw item to uniform output format (placeholder for dealing with these type of rows for now)")

    id = raw_not_null_verification_rollout_item['rollout_uuid']
    image = raw_not_null_verification_rollout_item['rollout_image_path']
    question_match = re.search(r'<question>(.*?)</question>', raw_not_null_verification_rollout_item['rollout_question'], re.DOTALL)
    question = question_match.group(1).strip() if question_match else None
    if question is None:
        logger.error(f"ERROR: No question found in rollout_question: {raw_not_null_verification_rollout_item['rollout_question']}")
        raise ValueError(f"ERROR: No question found in rollout_question: {raw_not_null_verification_rollout_item['rollout_question']}")
    full_rollout_response = raw_not_null_verification_rollout_item['rollout_response'] # we fetch this so we can label it based on the "first incorrect step" identified by the model_to_identify_first_incorrect_step
    steps_with_score = raw_not_null_verification_rollout_item['rollout_steps_with_score']
    
    verifier_identified_first_incorrect_step = tuple()
    if model_to_identify_first_incorrect_step == 'o4_mini':
        verification_solution = raw_not_null_verification_rollout_item['o4_mini_verification_solution']
        verifier_identified_first_incorrect_step = parse_first_correct_step_from_verification(verification_solution) # returns (first_incorrect_section_name, first_incorrect_step_num)  # 0-based
    else:
        logger.error(f"ERROR: Model {model_to_identify_first_incorrect_step} not supported")
        raise ValueError(f"ERROR: Model {model_to_identify_first_incorrect_step} not supported")

    # conversations = [{'from': 'system', 'value': PRM_SYSTEM_PROMPT}]
    # found_negative = False
    # first_incorrect_step = None

    # Now find the first incorrect step from the model verification trace
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
    

    conversations = [{'from': 'system', 'value': PRM_SYSTEM_PROMPT}]
    found_negative = False

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
                logger.error(f"Step solution not found in step_to_section_and_xml: {step_solution}")
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
        # if not found_negative and step['score'] <= threshold:
        # Need to normalize section names for comparison (remove brackets)
        current_section_normalized = current_section.replace('[', '').replace(']', '') if current_section else ''
        if not found_negative and ((current_section_normalized in ['Visual Elements', 'Perception'] and current_section_normalized == verifier_identified_first_incorrect_step[0] and (visual_elements_step_count - 1) == verifier_identified_first_incorrect_step[1]) or (current_section_normalized == 'Reasoning' and current_section_normalized == verifier_identified_first_incorrect_step[0] and (reasoning_step_count - 1) == verifier_identified_first_incorrect_step[1])):
            found_negative = True
        
        conversations.extend([
            {'from': 'human', 'value': step_solution},
            {'from': 'gpt', 'value': '-' if found_negative else '+'},
        ])

        # Early stop after processing the first negative step
        # if first step negative, then conversation only has 1 human-gpt value
        # for socres [0.8, 0.7, 0.6, -0.1, 0.5] and threshold 0.0, step 4 is the final step in the conversation
        if args.early_stop and found_negative:
            break

    return {
        'id': id,
        'image_url': image, # name follows process_vision_info qwen function requirement: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L321
        'conversations': conversations,
        'first_incorrect_step': None, # None cos we don't train on these samples
        'steps_with_score': steps_with_score,
        'consensus_filtering_algo_label': consensus_filtering_algo_label,
        'verifier_identified_first_incorrect_step_solution': verification_solution,
    } # return format: # final_mc_prm_data input df columns: (['id', 'image_url', 'conversations', 'first_incorrect_step', 'steps_with_score', "consensus_filtering_algo_label" -> "o4-mini_incorrect_and_MC_agrees_and_disagrees", "o4-mini_correct_and_MC_agrees", "o4-mini_correct_and_MC_disagrees"], "verifier_identified_first_incorrect_step_solution")


def mc_consensus_filtering_v2_algo(raw_not_null_verification_rollout_item: dict, all_items_array: list[dict]) -> dict:
    logger.debug(f"Running v2 consensus filtering algo on item: {raw_not_null_verification_rollout_item}")

    # we first separate o4-mini correct and incorrect samples
    o4_mini_correct_items = [item for item in all_items_array if item['o4_mini_isVerified']]
    o4_mini_incorrect_items = [item for item in all_items_array if not item['o4_mini_isVerified']]

    # verify that total number of items is the sum of correct and incorrect items
    if len(o4_mini_correct_items) + len(o4_mini_incorrect_items) != len(all_items_array):
        logger.error(f"ERROR: Total number of items is not the sum of correct and incorrect items")
        raise ValueError(f"ERROR: Total number of items is not the sum of correct and incorrect items")
    
    # we then check if the raw_not_null_verification_rollout_item is in o4-mini correct items
    if not (raw_not_null_verification_rollout_item in o4_mini_correct_items or raw_not_null_verification_rollout_item in o4_mini_incorrect_items):
        logger.debug(f"Raw item is not in o4-mini correct items or incorrect items")
        raise ValueError(f"ERROR: Raw item is in o4-mini correct items or incorrect items")
    else:
        logger.debug(f"Raw item is in o4-mini correct items or incorrect items, now let's check if MC agrees with o4-mini")

    # Now we check if this raw_not_null_verification_rollout_item is in o4-mini_correct_items AND has the MC threshold score agrees with o4-mini
    if raw_not_null_verification_rollout_item in o4_mini_correct_items:
        logger.debug(f"Passing Raw item through MC threshold filter first")
        # we then check if the MC threshold score agrees with o4-mini
        mc_filtered_item = item2conv_prm(raw_not_null_verification_rollout_item)
        if mc_filtered_item['first_incorrect_step'] is None:
            logger.debug(f"MC threshold and o4-mini agree on all steps correct")
            mc_filtered_item['consensus_filtering_algo_label'] = 'o4-mini_correct_and_MC_agrees'
            mc_filtered_item['verifier_identified_first_incorrect_step_solution'] = None
            # this group is 4)** MC and o4-mini agree on all steps correct
            logger.debug(f"returning mc_filtered_item with MC and o4-mini agree on all steps correct: {mc_filtered_item}")
            return mc_filtered_item
        else: # we assume o4-mini knows better than MC, identify first incorrect step based on o4-mini and output it in the same share_gpt format style mc_filtered_item before goes into final TRL filter. mc["first_incorrect_step"] is not None (found an incorrect step), o4-mini says its correct though. We don't train on these samples.
            logger.debug(f"MC threshold and o4-mini disagree on all steps correct. o4-mini thinks it is all correct, but MC results in incorrect answers.")
            logger.debug(f"mc_filtered_item has first incorrect step: {mc_filtered_item['first_incorrect_step']} but o4-mini thinks this trace is all correct. We don't train on these samples.")
            # this group is 3)** MC and o4-mini disagree
            o4_mini_correct_and_MC_disagrees_item = raw_item_to_uniform_output_format(raw_not_null_verification_rollout_item, 'o4_mini', 'o4-mini_correct_and_MC_disagrees')
            logger.debug(f"returning raw_item_to_uniform_output_format for debugging and error analysis later: {o4_mini_correct_and_MC_disagrees_item}")
            return o4_mini_correct_and_MC_disagrees_item

    elif raw_not_null_verification_rollout_item in o4_mini_incorrect_items:
        logger.debug(f"Raw item is in o4-mini incorrect items, processing item to first incorrect step identified by o4-mini")
        # we assume o4-mini knowns better, do not care if it agrees with MC or not, just return the item with the first incorrect step identified by o4-mini. This group is 1)** o4-mini incorrect, and MC agrees 2)** o4-mini incorrect, and MC disagrees
        o4_mini_incorrect_and_MC_agrees_and_disagrees_item = raw_item_to_model_identified_first_incorrect_step(raw_not_null_verification_rollout_item, 'o4_mini', 'o4-mini_incorrect_and_MC_agrees_and_disagrees')
        logger.debug(f"returning raw_item_to_model_identified_first_incorrect_step with o4-mini identified first incorrect step: {o4_mini_incorrect_and_MC_agrees_and_disagrees_item}")
        return o4_mini_incorrect_and_MC_agrees_and_disagrees_item
        
    # returns final_mc_prm_data input df columns: (['id', 'image_url', 'conversations', 'first_incorrect_step', 'steps_with_score', "consensus_filtering_algo_label" -> "o4-mini_incorrect_and_MC_agrees_and_disagrees", "o4-mini_correct_and_MC_agrees", "o4-mini_correct_and_MC_disagrees"], "verifier_identified_first_incorrect_step_solution")


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
            content.insert(0, {"type": "image", "image": convert_image_to_base64_string(final_mc_prm_data['image_url']), "index": 0}) # Load image from S3/image_url and add the base64 image.
            image_added = True
        
        trl_messages.append({
            "role": trl_role,
            "content": content
        })

    return {
        "messages": trl_messages,
        "images": [transform_image_url_to_s3(final_mc_prm_data['image_url'])],  # List of image URLs/paths
        "id": final_mc_prm_data['id']
    }


def check_data_integrity_of_convsprm(convs_prm, filtered_items):
    """
    Validate that all items in filtered_items are processed and added to convs_prm 
    with the expected consensus filtering labels.
    """
    # Collect and validate consensus filtering algo labels
    consensus_labels = []
    for item in convs_prm:
        if "consensus_filtering_algo_label" not in item:
            logger.error(f"ERROR: Item missing 'consensus_filtering_algo_label' field: {item['id']}")
            raise ValueError(f"ERROR: Item missing 'consensus_filtering_algo_label' field: {item['id']}")
        consensus_labels.append(item["consensus_filtering_algo_label"])

    # Get unique values
    unique_labels = set(consensus_labels)

    # Expected labels
    expected_labels = {
        "o4-mini_incorrect_and_MC_agrees_and_disagrees",
        "o4-mini_correct_and_MC_agrees", 
        "o4-mini_correct_and_MC_disagrees"
    }

    # Check that we have exactly the expected 3 unique values
    if unique_labels != expected_labels:
        logger.error(f"ERROR: Expected consensus filtering algo labels {expected_labels}, but got {unique_labels}")
        raise ValueError(f"ERROR: Expected consensus filtering algo labels {expected_labels}, but got {unique_labels}")

    # Count occurrences of each label
    from collections import Counter
    label_counts = Counter(consensus_labels)

    # Print counts for each unique value
    logger.info("Consensus filtering algo label counts:")
    total_count = 0
    for label in expected_labels:
        count = label_counts[label]
        logger.info(f"  {label}: {count}")
        total_count += count

    # Check that len(convs_prm) == len(filtered_items)
    if len(convs_prm) != len(filtered_items):
        logger.error(f"ERROR: len(convs_prm) ({len(convs_prm)}) != len(filtered_items) ({len(filtered_items)})")
        raise ValueError(f"ERROR: len(convs_prm) ({len(convs_prm)}) != len(filtered_items) ({len(filtered_items)})")

    # Additional validation that total counts match
    if total_count != len(filtered_items):
        logger.error(f"ERROR: Total label counts ({total_count}) != len(filtered_items) ({len(filtered_items)})")
        raise ValueError(f"ERROR: Total label counts ({total_count}) != len(filtered_items) ({len(filtered_items)})")

    logger.info(f"SUCCESS: All validation checks in check_data_integrity_of_convsprm passed. Processed {len(filtered_items)} items with {len(unique_labels)} unique consensus labels.")


def main():
    # log all configs:
    logger.info(f'{args=}')

    if not os.path.exists(args.data_dir):
        logger.error(f'Dir does not exist: {args.data_dir}')
        exit(0)

    # Get list of files to process
    # files_to_process = [f for f in os.listdir(args.data_dir) if f.endswith('.jsonl')]
    files_to_process = [
        'vqav2_final_mc_rollouts_with_all_models_verification_merged.jsonl', 
        'InfoVQA_final_mc_rollouts_with_all_models_verification_merged.jsonl', 
        'CLEVR_final_mc_rollouts_with_all_models_verification_merged.jsonl', 
        'RAVEN_final_mc_rollouts_with_all_models_verification_merged.jsonl', 
        'dvqa_final_mc_rollouts_with_all_models_verification_merged.jsonl', 
        'AI2D_final_mc_rollouts_with_all_models_verification_merged.jsonl'
    ]
    
    for filename in tqdm(files_to_process, desc="Processing files", unit="file"):
        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, f'train/mc{args.mc_threshold}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, f'debug/mc{args.mc_threshold}'), exist_ok=True)

        pairs_save_path = os.path.join(save_dir, f'debug/mc{args.mc_threshold}', f'{ds_name}_prm_training_data_mc{args.mc_threshold}.jsonl')
        final_trl_format_save_path = os.path.join(save_dir, f'train/mc{args.mc_threshold}', f'{ds_name}_prm_training_data_final_trl_format_mc{args.mc_threshold}.jsonl')
        # pairs_orm_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_orm.jsonl')

        if os.path.exists(pairs_save_path) and not args.overwrite:
            logger.info(f'Debug File already exists: {pairs_save_path} or Train file path already exists: {final_trl_format_save_path}')
            logger.info(f'Skipping file: {filename}')
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
                logger.debug(f"Skipping item because it has None values in verification columns")
                continue
            filtered_items.append(item)

        # Core filtering logic comes here:
        processed_count = 0
        skipped_count = 0
        
        progress_bar = tqdm(filtered_items, desc=f"Processing {ds_name}", unit="sample", leave=False)
        for item in progress_bar:
            if args.consensus_filtering_algo_version == 'v1':
                logger.debug(f'Running v1 consensus filtering algo on new item')
                # First, we apply the MC threshold to the item and get the "first incorrect step" based on the threshold
                mc_filtered_item = item2conv_prm(item)
                # add a function to check if mc_filtered_item has first step incorrect. If so, then we can skip the item.
                if is_LLM_judge_consensus_filtering(mc_filtered_item, filtered_items): # TODO (if using v1): collect IDs here to count and check manually
                    convs_prm.append(mc_filtered_item)
                    final_filtered_item = final_filter_and_processing_before_training(mc_filtered_item)
                    final_trl_format_items.append(final_filtered_item)
                    processed_count += 1
                else:
                    skipped_count += 1
                    continue # track rows that failed the consensus filtering in another array that is saved for auditing
                # TODO: collect IDs here to count and check manually
            elif args.consensus_filtering_algo_version == 'v2':
                logger.debug(f'Running v2 consensus filtering algo on new item')
                mc_consensus_filtered_v2_item = mc_consensus_filtering_v2_algo(item, filtered_items) # expected output schema: (['id', 'image_url', 'conversations', 'first_incorrect_step', 'steps_with_score'])

                convs_prm.append(mc_consensus_filtered_v2_item)

                if mc_consensus_filtered_v2_item is not None and mc_consensus_filtered_v2_item['consensus_filtering_algo_label'] != "o4-mini_correct_and_MC_disagrees":
                    final_filtered_item = final_filter_and_processing_before_training(mc_consensus_filtered_v2_item)
                    final_trl_format_items.append(final_filtered_item)
                    processed_count += 1
                else:
                    skipped_count += 1
            else:
                logger.error(f"ERROR: Invalid consensus filtering algo version: {args.consensus_filtering_algo_version}")
                raise ValueError(f"ERROR: Invalid consensus filtering algo version: {args.consensus_filtering_algo_version}")
 
            
            statistics['num_turns'].append(len(convs_prm[-1]['conversations']))
            
            # Update progress bar with current stats
            progress_bar.set_postfix({
                'processed': processed_count,
                'skipped': skipped_count,
                'total_output': len(final_trl_format_items)
            })

        # Validate data integrity of convs_prm
        check_data_integrity_of_convsprm(convs_prm, filtered_items)
        
        # Log final processing statistics
        logger.info(f'[{filename}] Processing Summary:')
        logger.info(f'  Total items loaded: {len(items)}')
        logger.info(f'  Items after None filtering: {len(filtered_items)}')
        logger.info(f'  Items processed successfully: {processed_count}')
        logger.info(f'  Items skipped: {skipped_count}')
        logger.info(f'  Final training items: {len(final_trl_format_items)}')
        logger.info(f'  Debug items (convs_prm): {len(convs_prm)}')
        
        for k, v in info.items():
            logger.info(f'{k}: {v}')
        for k, v in statistics.items():
            logger.info(f'{k}: max={max(v)}, min={min(v)}, mean={sum(v) / len(v)}, total={sum(v)}')
        logger.info('')

        save_outputs(convs_prm, pairs_save_path)
        save_outputs(final_trl_format_items, final_trl_format_save_path)
        # if args.include_orm_data:
            # save_outputs(convs_orm, pairs_orm_save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/fast10/brandon/mmr_rollout_data/final_combined_MC_and_verification_files_updated_rollouts') # ran InfoVQA and AI2D on the base final_combined_MC_and_verification_files without updating the o4_mini_isVerified from False to None for verification_solutions that are missing Section Headers
    parser.add_argument('--save-dir', type=str, default='/mnt/fast10/brandon/mmr_rollout_data/prm_training_data')
    parser.add_argument('--mc-threshold', type=float, default=0.0) # TODO: try 0.5 and 0.8; and maybe include/exclude nano. Point is to find more "-" points where LLM Judge can agree on it being an error. (0.8 comes from GenPRM recommendation for math reasoning)
    parser.add_argument('--early-stop', action='store_true', default=True)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--consensus-filtering-algo-version', type=str, default='v2') 
    # parser.add_argument('--include-orm-data', action='store_true', default=False)
    args = parser.parse_args()

    main()
    # usage: python convert_mc_to_prm_signal.py 
    # all default values should work