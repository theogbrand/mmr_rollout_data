1. First we flatten and merge all the verification query (sent for batch verification), and all the verification results (received from the batch API) into a single file.
    - there will be equal number of verification queries as rollouts (25557 for AI2D), but the verification results will be less than that, because some of the queries were error-ed out when we ran out of credits for the batch API.
    - the verification results become the "limiting factor" for the number of valid rollouts we can use for training.
    - Refer to ```flatten_rollout_and_verification_files.ipynb``` for more details.

2. 
    a. Then we merge the flattened verification queries with the verification results into a single file, using "custom_id" as the intersection key.
        - Here, we used verification query files as the "reference point", since it turns out for each model (GPT-4.1-mini/nano, o4-mini), have different number of successful verification results received from the batch API.
        - Also, as mentioned in #1, there will be fewer verification results than queries, so we set queries with no verification results as "none" for the new field {model_name}_isVerified.
            - This field is a boolean field, which is determined based on first parsing the response from the verification response text between <conclusion> and </conclusion> tags.
            - If extracted text is "Correct" (case insensitive), then the field is set to True, otherwise False.
            - Especially for smaller models like GPT-4.1-mini/nano, the model more likely fails to follow the verification instructions closely and commonly regurgigates trying to answer the question or gives a long chatty response before verifying. We consider these cases as "invalid" and set the field to None. 
        - Refer to first half of ```process_verification_files.ipynb``` and ```process_verification_files.py``` for more details.
        - ```process_verification_files.py``` is the script that merges the verification query and results files. Which gives us 3 files per model, and if model has subset, in merged_verification_files folder, there will be 3 files per model per subset. (CLEVR has 2 subsets so 6 files total - 2 per model)
        - there will be a "verification_merged" file per model per section, and a "final_verification_processed" file per model per section.
            - Merge ("stack") subset files into 1 file to 3 "model-level" files before can finally merge with the rollout file in step 2B below

3   a. Finally we merge the merged verification and results file (in merged_verification_files folder) with the rollout files (in flattened_rollout_files folder) using "response" text as the intersection key.
        - We made an error here during verification generation, and was supposed to use the rollout "uid" as the "custom_id", and then use that as the intersection key. (which we will update in the next round of rollouts in the verification generation pipeline)
        - But luckily we can still use the "response" text as the intersection key, since the response text should be unique for each rollout. We check for collisions first, to debug and determine a way to merge later.
        - so far we defer this fix to later since there are no collisions. Then we merge the rollout files with the merged verification and results file, using the rollout file as the "reference point", and setting the {model_name}_isVerified field to a value of None if there are no corresponsing verification results for that rollout.
        - Refer to ```merge_rollout_and_verification_files.ipynb``` for more details.
        - At this point every model is separately merged with the rollout files, and we have 3 files for each dataset, so we need to merge this into 1 file, taking the rollout file as the "reference point".

    b. We merge all three model verification files into a single file, taking the rollout file as the "reference point".
    - Refer to ```merge_all_model_verification_files.ipynb``` for more details
        - run ```pyton merge_rollout_and_verification_files.py test``` to test the merge function on ONE dataset of prompts first (AI2D).
        - to test the multi dataset version
        - to test outputs from ```python convert_mc_to_prm_signal.py``` and verify first incorrect step is correct

    b. We use ```convert_mc_to_prm_signal.py``` which takes in a threshold value, converts stepwise scores into "+/-" PRM signal, and filters out rollouts with consensus between the three model verification results
