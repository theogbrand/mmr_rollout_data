1. First we flatten and merge all the verification query (sent for batch verification), and all the verification results (received from the batch API) into a single file.
    - there will be equal number of verification queries as rollouts (25557 for AI2D), but the verification results will be less than that, because some of the queries were error-ed out when we ran out of credits for the batch API.
    - the verification results become the "limiting factor" for the number of valid rollouts we can use for training.
    - Refer to ```flatten_rollout_and_verification_files.ipynb``` for more details.

2. Then we merge the flattened verification queries with the verification results into a single file, using "custom_id" as the intersection key.
    - Here, we used verification query files as the "reference point", since it turns out for each model (GPT-4.1-mini/nano, o4-mini), have different number of successful verification results received from the batch API.
    - Also, as mentioned in #1, there will be fewer verification results than queries, so we set queries with no verification results as "none" for the new field {model_name}_isVerified.
        - This field is a boolean field, which is determined based on first parsing the response from the verification response text between <conclusion> and </conclusion> tags.
        - If extracted text is "Correct" (case insensitive), then the field is set to True, otherwise False.
        - Especially for smaller models like GPT-4.1-mini/nano, the model more likely fails to follow the verification instructions closely and commonly regurgigates trying to answer the question or gives a long chatty response before verifying. We consider these cases as "invalid" and set the field to None. 
        - Refer to ```process_verification_files.ipynb``` and ```process_verification_results.py``` for more details.

