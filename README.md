1. Goal: Flatten and merge all the verification query (sent for batch verification), and all the verification results (received from the batch API) into a single file. (at a model and optonal dataset subset level)
    a. First we flatten and merge all the verification query (sent for batch verification), and all the verification results (received from the batch API) into a single file.
    - there will be equal number of verification queries as rollouts (25557 for AI2D), but the verification results will be less than that, because some of the queries were error-ed out when we ran out of credits for the batch API.
    - the verification results become the "limiting factor" for the number of valid rollouts we can use for training.
    - Refer to ```flatten_rollout_and_verification_files.ipynb``` (main file to run) for more details.
    - Creates a {dataset_name}_flattened.jsonl file in flattened_rollout_files folder.

2. Goal: Merge the flattened verification queries with the verification results into a single file, before merging with the rollout file.
    a. Then we merge the flattened verification queries with the verification results into a single file, using "custom_id" as the intersection key.
        - Here, we used verification query files as the "reference point", since it turns out for each model (GPT-4.1-mini/nano, o4-mini), have different number of successful verification results received from the batch API.
        - Also, as mentioned in #1, there will be fewer verification results than queries, so we set queries with no verification results as "none" for the new field {model_name}_isVerified.
            - This field is a boolean field, which is determined based on first parsing the response from the verification response text between <conclusion> and </conclusion> tags.
            - If extracted text is "Correct" (case insensitive), then the field is set to True, otherwise False.
            - Especially for smaller models like GPT-4.1-mini/nano, the model more likely fails to follow the verification instructions closely and commonly regurgigates trying to answer the question or gives a long chatty response before verifying. We consider these cases as "invalid" and set the field to None. 
        - "stack subset" workflow only applies if within the verification_pipeline_outputs folder, there are multiple subsets for the same dataset. (e.g. RAVEN has 7 subsets, CLEVR has 2 subsets, dvqa has 3 subsets)
        - Refer to ```process_verification_files.py``` (main file to run) to combine the verification query and results files.
        - ```process_verification_files.py``` is the script that merges the verification query and results files. Which gives us 3 files per model, and if model has subset, in merged_verification_files folder, there will be 3 files per model per subset. (CLEVR has 2 subsets so 6 files total - 2 per model)
        - there will be a "verification_merged" file (just the combined query and results files) per model per section, and a "final_verification_processed" file (we parse the response text to get the conclusion if it is correct or incorrect) per model per section.
            - Merge ("stack") subset files into 1 file to 3 "model-level" files before can finally merge with the rollout file in step 2B below. usethe first part of ```merge_rollout_and_verification_files.ipynb``` to merge the verification query and results files.

3. Goal: Combine the merged verification and results file (in merged_verification_files folder) with the rollout files (in flattened_rollout_files folder) using "response" text as the intersection key.
        - Refer to ```merge_rollout_and_verification_files.ipynb``` to first ensure datasets with multiple sections are merged into 1 for more details, then proceed to run ```python merge_rollout_and_verification_files.py test``` to merge the rollout files with the merged verification and results file.
            - ```test``` is a flag to run the test function, which is currently only tested for a single dataset (level) so we can check pipeline for data integrity.
        - We made an error here during verification generation, and was supposed to use the rollout "uid" as the "custom_id", and then use that as the intersection key. (which we will update in the next round of rollouts in the verification generation pipeline)
        - But luckily we can still use the "response" text as the intersection key, since the response text should be unique for each rollout. We check for collisions first, to debug and determine a way to merge later.
        - so far we defer this fix to later since there are no collisions. Then we merge the rollout files with the merged verification and results file, using the rollout file as the "reference point", and setting the {model_name}_isVerified field to a value of None if there are no corresponsing verification results for that rollout.
        - Refer to ```merge_rollout_and_verification_files.ipynb``` for more details.
        - At this point every model is separately merged with the rollout files, and we have 3 files for each dataset, so we need to merge this into 1 file, taking the rollout file as the "reference point".

    b. We merge all three model verification files into a single file, taking the rollout file as the "reference point".
    - Look at distribution of LLM Judge agreement using ```merge_rollout_and_verification_files.ipynb```
    - Refer to ```merge_all_model_verification_files.ipynb``` for more details
        - run ```pyton merge_rollout_and_verification_files.py test``` to test the merge function on ONE dataset of prompts first (AI2D).
        - to complete and test the multi dataset version

4.  run ```drop_problematic_rollouts.py``` to drop problematic rollouts that do not have proper section headers and steps denoted (inclusive of those with verified correct AND incorrect), and ```update_problematic_verification_solutions.py``` to update the o4_mini_isVerified field to None for problematic verification solutions, usually when the o4_mini_isVerified field is False but the verification_solution does not output the expected strict XML format for us to parse the first incorrect step properly
    - run first without saving and refer to ```edit_final_combined_MC_and_verification_files_manually.ipynb``` to trace and verify the updated solutions

TODO:
- rerun the first 3 (in completed_prm_training_data_conversion folder which should be the same with the raw final_combined_MC_and_verification_files folder) with drop and update steps in #4. Right after the last #3 are done (with update and drop steps - sequence shouldn't matter)

5. Goal: Finalize data labeling algorithm (Correct: MC+o4-mini agree, where MC Threshold > 0 considered correct step, and o4-mini agrees trace is all correct; Incorrect: o4-mini first incorrect step ONLY)
    - We use ```convert_mc_to_prm_signal.py``` which takes in a threshold value, converts stepwise scores into "+/-" PRM signal, and filters out rollouts with consensus between the three model verification results

6. Run all cells in ```getStats_and_flatten_final_prm_training_data.ipynb``` to get statistics and flatten the final prm training data into a single JSONL file in the format ready for TRL

7. Run ```convert_arrow.py``` to convert the data to arrow format shards before pushing to HuggingFace.
    - Note, now we run process_vision_info for Qwen at Training time, in the collate_fn function and use it in the trainer, so we do not need to process the image at this stage
    - run ```push_to_hf.py``` to push the data to HuggingFace.