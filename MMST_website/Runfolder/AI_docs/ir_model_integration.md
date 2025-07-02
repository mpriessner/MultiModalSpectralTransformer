# IR Model Integration Analysis

## Overview

The IR (Infrared) spectroscopy simulation functionality is a critical component of the MultiModalSpectralTransformer website. This document analyzes how the IR model is integrated into the website and identifies potential issues that may be causing problems.

## Directory Structure Requirements

According to the project configuration, the IR model files are expected to be located at:

```
models/chemprop-ir/ir_models_data/experiment_model/model_files/
```

This directory structure is referenced in the `ir_config_V8.json` file:

```json
{   
    "checkpoint_dir": ["../../models/chemprop-ir/ir_models_data/experiment_model/model_files"], 
    // Other configuration parameters...
}
```

## IR Model Integration Flow

### 1. Configuration Loading

The IR model configuration is loaded in `MMT_import.py`:

```python
def load_configs():
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build paths relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, ''))
    
    ir_config_path = os.path.join(base_path, 'ir_config_V8.json')
    config_path = os.path.join(base_path, 'config_V8.json')
    
    # Load IR config
    IR_config_dict = load_config(ir_config_path)
    if IR_config_dict is None:
        raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
    IR_config = parse_arguments(IR_config_dict)
    modify_predict_args(IR_config)
    
    # Load main config
    config_dict = load_config(config_path)
    if config_dict is None:
        raise FileNotFoundError(f"Main config file not found at {config_path}")
    config = parse_arguments(config_dict)
 
    return IR_config, config
```

### 2. Simulation Process

The IR simulation is triggered by the `/simulate` route in `app.py`:

```python
@app.route('/simulate/<path:SMILES_Path>', methods=['GET'])
def simulate(SMILES_Path):
    try:
        print_to_console("Function simulate: Start of Simulation")
        print(f"SMILES_Path: {SMILES_Path}")
        
        # Check if the SMILES file exists
        if not os.path.exists(SMILES_Path):
            error_msg = f"SMILES file not found at path: {SMILES_Path}"
            print_to_console(error_msg)
            print(error_msg)
            return jsonify({"error": error_msg}), 404
            
        IR_config, config = load_configs()
        config.simulated = True
        config.SGNN_csv_gen_smi = "/" + SMILES_Path
        save_updated_config(config, config.config_path)
        
        try:
            config = sim_and_display()  # Actual simulation call
            print_to_console("Function simulate: Simulation Succeeded")
            return '', 204
        except Exception as e:
            error_msg = f"Error during simulation: {str(e)}"
            print_to_console(error_msg)
            print(error_msg)
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            print_to_console(traceback_str)
            return jsonify({"error": error_msg, "traceback": traceback_str}), 500
    # Exception handling...
```

### 3. Actual Simulation

The actual IR simulation happens in the `sim_and_display` function in `MMT_import.py`:

```python
def sim_and_display():
    print("="*50)
    print("sim_and_display function called")
    try:
        print("Loading dictionaries...")
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        print("Loading configs...")
        IR_config, config = load_configs()
        
        print(f"SGNN_csv_gen_smi path: {config.SGNN_csv_gen_smi}")
        
        # Ensure the path doesn't have a leading slash when checking file existence
        csv_path = config.SGNN_csv_gen_smi
        if csv_path.startswith('/'):
            csv_path = csv_path[1:]
            
        config.csv_SMI_targets = config.SGNN_csv_gen_smi
        
        # Check if the file exists
        print(f"Checking if file exists: {csv_path}")
        if not os.path.exists(csv_path):
            print(f"ERROR: File not found at path: {csv_path}")
            raise FileNotFoundError(f"File not found at path: {csv_path}")
        else:
            print(f"File exists: {csv_path}")
            
        # Check if the file is readable
        try:
            with open(csv_path, 'r') as f:
                first_line = f.readline()
                print(f"First line of CSV: {first_line}")
        except Exception as e:
            print(f"ERROR reading file: {str(e)}")
            raise
            
        print("Cleaning dataset...")
        config = ex.clean_dataset(config)
        print("Dataset cleaned successfully")
        
        print("\033[1m\033[31mThis is: simulate_syn_data\033[0m")
        
        print("Generating simulated data...")
        print(f"IR_config checkpoint_dir: {IR_config.checkpoint_dir}")
        print(f"IR_config test_path: {IR_config.test_path}")
        print(f"IR_config preds_path: {IR_config.preds_path}")
        
        # Check if IR model files exist
        if not os.path.exists(IR_config.checkpoint_dir):
            print(f"WARNING: IR checkpoint directory not found: {IR_config.checkpoint_dir}")
        else:
            print(f"IR checkpoint directory exists: {IR_config.checkpoint_dir}")
            # List files in the directory
            print("Files in checkpoint directory:")
            for file in os.listdir(IR_config.checkpoint_dir):
                print(f"  - {file}")
        
        config = ex.gen_sim_aug_data(config, IR_config) 
        print("Simulated data generated successfully")

        print("Setting display paths...")
        config.csv_1H_path_display = config.csv_1H_path_SGNN
        config.csv_13C_path_display = config.csv_13C_path_SGNN
        config.csv_HSQC_path_display = config.csv_HSQC_path_SGNN
        config.csv_COSY_path_display = config.csv_COSY_path_SGNN
        config.IR_data_folder_display = config.IR_data_folder
        
        print(f"1H path: {config.csv_1H_path_display}")
        print(f"13C path: {config.csv_13C_path_display}")
        print(f"HSQC path: {config.csv_HSQC_path_display}")
        print(f"COSY path: {config.csv_COSY_path_display}")
        print(f"IR folder: {config.IR_data_folder_display}")
        
        save_updated_config(config, config.config_path)
        print("sim_and_display completed successfully")
        print("="*50)
        return config
    except Exception as e:
        print(f"ERROR in sim_and_display: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("="*50)
        raise
```

### 4. IR Data Visualization

The IR data is visualized through the `/plot_nmr` route in `app.py`:

```python
@app.route('/plot_nmr')
def plot_nmr():
    try:
        # ...
        elif nmr_type == "IR":
            IR_data = {}
            NMR_file = get_path(config_data, "1H")
            data_dict = pd.read_csv(NMR_file)
            sample_ids = list(data_dict["sample-id"])
            current_sample_id = sample_ids[index]
            IR_folder = get_path(config_data, nmr_type)
            IR_csv_path = os.path.join(IR_folder, str(current_sample_id) + ".csv")            
            IR_df = pd.read_csv(IR_csv_path)

            if IR_df.shape[1] != 1:
                logger.error(f"Invalid column count for IR data in {IR_csv_path}")
                return
            absorbance = IR_df.iloc[:, 0].astype(float).tolist()
            wave_lengths = np.linspace(400, 4000, len(absorbance))
            IR_data = {'wave_lengths': wave_lengths, 'absorbance': absorbance}
        # ...
        elif nmr_type == 'IR':
            fig.add_trace(go.Scatter(x=wave_lengths, y=absorbance, mode='lines', name='IR Spectrum', line=dict(color='red')))
            fig.update_layout(
                title='IR Spectrum',
                xaxis=dict(title='Wavenumber (cm⁻¹)'),
                yaxis=dict(title='Absorbance')
            )
        # ...
    except Exception as e:
        logger.error(f"Error in plot_nmr: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500
```

## Identified Issues

### 1. Missing IR Model Directory Structure

**Issue**: The required directory structure for IR models is missing:
```
models/chemprop-ir/ir_models_data/experiment_model/model_files/
```

**Impact**: This will cause the IR simulation to fail because the model files cannot be found.

**Error Symptoms**:
- Error messages about missing model files
- Failed simulation attempts
- Empty IR spectra

**Solution**:
1. Create the required directory structure:
   ```
   mkdir -p models/chemprop-ir/ir_models_data/experiment_model/model_files/
   ```
2. Obtain the necessary IR model files and place them in the `model_files` directory
3. Ensure the paths in `ir_config_V8.json` correctly point to this directory

### 2. Path Resolution in Windows Environment

**Issue**: Path resolution in Windows can be problematic, especially with mixed forward and backward slashes.

**Impact**: Paths may not resolve correctly, leading to "file not found" errors.

**Error Symptoms**:
- File not found errors despite files existing
- Path-related exceptions

**Solution**:
1. Use `os.path.join()` consistently for path construction
2. Avoid hardcoded path separators
3. Use absolute paths where possible
4. Normalize paths with `os.path.normpath()`

### 3. Conda Environment Activation Issues

**Issue**: When activating Conda environments in Windows batch scripts, using full paths can cause issues.

**Impact**: Python modules may not be found even if they are installed.

**Error Symptoms**:
- ModuleNotFoundError exceptions
- Import errors

**Solution**:
1. Use `conda activate <env_name>` instead of `conda activate <full_path_to_env>`
2. Initialize the base Conda environment first with `call <path_to_conda>\Scripts\activate.bat`
3. Verify the environment is correctly activated before running Python scripts

## Testing IR Model Integration

To test if the IR model integration is working correctly:

1. **Check Model Files**:
   ```python
   import os
   
   model_dir = "models/chemprop-ir/ir_models_data/experiment_model/model_files/"
   if os.path.exists(model_dir):
       print(f"Model directory exists: {model_dir}")
       print("Files in directory:")
       for file in os.listdir(model_dir):
           print(f"  - {file}")
   else:
       print(f"Model directory does not exist: {model_dir}")
   ```

2. **Test IR Config Loading**:
   ```python
   from MMT_import import load_configs
   
   try:
       IR_config, config = load_configs()
       print(f"IR config loaded successfully")
       print(f"IR_config checkpoint_dir: {IR_config.checkpoint_dir}")
       print(f"IR_config test_path: {IR_config.test_path}")
   except Exception as e:
       print(f"Error loading IR config: {str(e)}")
   ```

3. **Test IR Simulation with a Simple SMILES**:
   ```python
   # Create a test SMILES file
   import os
   
   test_dir = "test_data"
   os.makedirs(test_dir, exist_ok=True)
   
   test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
   with open(os.path.join(test_dir, "test_smiles.csv"), "w") as f:
       f.write("SMILES\n")
       f.write(f"{test_smiles}\n")
   
   # Test simulation
   from MMT_import import sim_and_display
   
   try:
       config.SGNN_csv_gen_smi = os.path.join(test_dir, "test_smiles.csv")
       config = sim_and_display()
       print("Simulation successful")
   except Exception as e:
       print(f"Simulation failed: {str(e)}")
   ```

## Conclusion

The IR model integration is a critical component of the MultiModalSpectralTransformer website. The main issues identified are:

1. Missing model directory structure
2. Path resolution problems in Windows
3. Conda environment activation issues

By addressing these issues, the IR simulation functionality should work correctly, allowing the website to generate and display IR spectra for the provided SMILES structures.
