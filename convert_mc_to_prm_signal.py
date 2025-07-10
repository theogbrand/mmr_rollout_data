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
    id = item['response_uid']
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


def find_index_of_first_incorrect_step_in_verification_trace(mc_filtered_item, all_items_array):
    return

def check_all_step_correct_consensus(mc_filtered_item, all_items_array):
    return

def is_LLM_judge_consensus_filtering(mc_filtered_item, all_items_array):
    # if mc filtered item is correct
    if mc_filtered_item['first_incorrect_step'] is None:
        # just need to check if {model_name}_isVerified is True for all models
            # if check is True, then return True
        return check_all_step_correct_consensus(mc_filtered_item, all_items_array)

    else: # check if first incorrect step is the same for verification traces
        return find_index_of_first_incorrect_step_in_verification_trace(mc_filtered_item, all_items_array)
    
    return


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

        # for item in items:
        #     image = item['image_path']
        #     question = item['question']
        #     steps_with_score = item['steps_with_score']

        #     score = steps_with_score[-1]['score']
        #     id2scores[(str(image), question)].append(score)

        for item in items:
            mc_filtered_item = convs_prm.append(item2conv_prm(item))
            if is_LLM_judge_consensus_filtering(mc_filtered_item, items):
                final_filtered_item = final_filter_and_processing_before_training(mc_filtered_item)
            else:
                continue # track rows that failed the consensus filtering in another array that is saved for auditing
            
            convs_prm.append(final_filtered_item)
 
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