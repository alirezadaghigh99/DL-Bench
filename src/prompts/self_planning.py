def self_planning_plan(prompt):
    dl_bench_plan = f'''
    Intent: Generate a Python function called build_charlm_vocab that builds a vocabulary for a CharacterLanguageModel. 
    The function takes in a file path and an optional cutoff value. It reads the files in the specified path, accumulates the counter of characters, 
    and then passes a list of characters to the vocab builder. The function removes infrequent characters based on the cutoff value.
    If the training data is empty or all characters are less frequent than the cutoff, it raises a ValueError. The function returns the built vocabulary.

    Plan: Let's think step by step.

    1. Prepare a Counter to track character frequencies.
    2. Identify filenames (handling both a directory or a single file).
    3. Update the Counter by reading each file line by line.
    4. Remove characters below the cutoff; if none remain, raise an error.
    5. Build and return the CharVocab from the remaining characters.
    
    Intent: Create a Python function _jpeg_encode that performs JPEG encoding on a batch of RGB images. The function takes the following parameters:

image_rgb: A tensor of shape (B, 3, H, W) representing a batch of RGB images.
jpeg_quality: A tensor of shape (B) representing the JPEG compression quality for each image in the batch.
quantization_table_y: A tensor representing the quantization table for the Y (luminance) channel.
quantization_table_c: A tensor representing the quantization table for the Cb and Cr (chrominance) channels.
The function returns a tuple of three tensors:

y_encoded: A tensor of shape (B, N, 8, 8) representing the encoded Y component.
cb_encoded: A tensor of shape (B, N, 8, 8) representing the encoded Cb component.
cr_encoded: A tensor of shape (B, N, 8, 8) representing the encoded Cr component.

    Plan: Let's think step by step.
    1. Convert the batch of RGB images to YCbCr.
    2. Scale the pixel values from [0, 1] to [0, 255].
    3. Chroma subsample the YCbCr image.
    4. Patchify the Y, Cb, and Cr channels into 8×8 blocks.
    5. Apply DCT to each 8×8 block.
    6. Quantize the DCT coefficients using jpeg_quality and the respective quantization tables.
    7. Return the quantized Y, Cb, and Cr tensors.

    Intent: "Generate a Python function called _compute_label_quality_scores that takes in the following parameters:
- labels: a list of dictionaries containing any type of values
- predictions: a list of numpy arrays
- method: an optional string parameter with a default value of ""objectlab""
- aggregation_weights: an optional dictionary with string keys and float values
- threshold: an optional float parameter
- overlapping_label_check: an optional boolean parameter with a default value of True
- verbose: a boolean parameter with a default value of True

The function prunes extra bounding boxes and computes label quality scores based on the specified method. If the method is ""objectlab"", it calculates the scores using specific parameters. Otherwise, it raises a ValueError.

The function returns a numpy array of computed scores."
    Plan: Let's think step by step.

Parse and Prepare Inputs

Accept labels (list of dicts) and predictions (list of numpy arrays).
Extract or set defaults for optional parameters (method, aggregation_weights, threshold, overlapping_label_check, verbose).
Prune Predictions

Get the minimum prediction probability (min_pred_prob).
If threshold is provided, prune predictions by that threshold.
Otherwise, set threshold to min_pred_prob.
Choose Method

Check if method equals "objectlab".
If it does, call _get_subtype_label_quality_scores with the relevant parameters (including aggregation_weights and overlapping_label_check).
Otherwise, raise a ValueError indicating the method is invalid.
Return Scores

Return the computed numpy array of label quality scores.
    How about this intent: {prompt}.

    Plan: Let's think step by step.
    '''
    
    return dl_bench_plan


def self_planning_implementation(prompt, plan):
    dl_bench_code = f'''
    {prompt}
    Please complete the task with the following steps in Python.

    {plan}
    '''
    return dl_bench_code




    
