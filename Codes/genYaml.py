import yaml
import os
import re
import datetime 

def load_hyperparameters(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params

work_folder = f'ParamsFolder'
# if it does not exist, create the folder
os.makedirs(work_folder, exist_ok=True)
def sanitize_for_filename(value):
    """Converts a string to a safe filename format by replacing non-alphanumeric characters."""
    return re.sub(r'[^a-zA-Z0-9]', '_', value)

def generate_filename(params):
    # Format ds as "64x128" for the filename
    ds_str = "x".join(map(str, params.get("ds", [])))
    # Sanitize sigma codes for safe inclusion in filenames
    sigma1_str = sanitize_for_filename(params['sigma1_code'])
    sigma2_str = sanitize_for_filename(params['sigma2_code'])
    FlagOverlapLin_str = f"FlagOverlapLin{params['FlagOverlapLin']}"
    FlagOverlapNonLin_str = f"FlagOverlapNonLin{params['FlagOverlapNonLin']}"
    FlagTrain2Layer_str = f"FlagTrain2Layer{params['FlagTrain2Layer']}"
    coef_iter_str = f"coefIter{params['coef_iter']}"
    fraction_batch_str = f"fractionBatch{params['fraction_batch']}"
    # Get the current time as a string
    time_creation = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Combine all parts to create a descriptive filename adding the time of creation at the end
    filename = f"{sigma1_str}_{sigma2_str}_ds{ds_str}_{FlagOverlapLin_str}_{FlagOverlapNonLin_str}_{FlagTrain2Layer_str}_{coef_iter_str}_{fraction_batch_str}_{time_creation}.yaml"
    return filename

def add_filename_to_params(params):
    # Generate a descriptive filename and add it as a field to the params
    generated_filename = generate_filename(params)
    params["filename"] = generated_filename
    return params

def process_all_yaml_files(raw_dir, clean_dir):
    # Ensure the clean directory exists
    os.makedirs(clean_dir, exist_ok=True)

    # Process each YAML file in the raw directory
    for filename in os.listdir(raw_dir):
        if filename.endswith('.yaml'):
            raw_filepath = os.path.join(raw_dir, filename)
            params = load_hyperparameters(raw_filepath)

            # Add the generated filename field to params
            clean_params = add_filename_to_params(params)

            # Determine the new file path in the clean directory
            clean_filename = clean_params["filename"]
            clean_filepath = os.path.join(clean_dir, clean_filename)

            # Save the modified YAML to the clean directory
            with open(clean_filepath, 'w') as file:
                yaml.dump(clean_params, file)

            print(f"Processed and saved: {clean_filepath}")

# Directories for raw and clean YAML files
raw_hyperparams_dir = 'RawHyperparams'
# clean_hyperparams_dir = 'hyperparams'
clean_hyperparams_dir = work_folder

# Run the process
process_all_yaml_files(raw_hyperparams_dir, clean_hyperparams_dir)
