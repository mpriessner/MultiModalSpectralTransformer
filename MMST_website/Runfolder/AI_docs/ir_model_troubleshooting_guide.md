# IR Model Troubleshooting Guide

This guide provides step-by-step instructions for fixing the IR model integration issues in the MultiModalSpectralTransformer website.

## Prerequisites

- Access to the MultiModalSpectralTransformer repository
- Basic understanding of Python and file system operations
- Access to the IR model files (if not already available)

## Issue Summary

The IR (Infrared) spectroscopy simulation functionality is failing due to:

1. Missing IR model directory structure
2. Path resolution issues in Windows environment
3. Conda environment activation problems

## Step 1: Create Required Directory Structure

The IR model files are expected to be in a specific directory structure that is currently missing:

```
models/chemprop-ir/ir_models_data/experiment_model/model_files/
```

### Instructions:

1. Navigate to the root of the MultiModalSpectralTransformer repository
2. Create the required directory structure:

```batch
mkdir -p models\chemprop-ir\ir_models_data\experiment_model\model_files
```

## Step 2: Obtain and Place IR Model Files

You need to obtain the IR model files and place them in the correct directory.

### Instructions:

1. Obtain the IR model files (from your team or project repository)
2. Copy all model files to:
   ```
   models\chemprop-ir\ir_models_data\experiment_model\model_files\
   ```
3. Verify the files are correctly placed:
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

## Step 3: Fix Path Resolution Issues

Path resolution in Windows can be problematic due to mixed forward and backward slashes.

### Instructions:

1. Open `ir_config_V8.json` in the `MMST_website\Runfolder` directory
2. Verify the checkpoint directory path:
   ```json
   {
       "checkpoint_dir": ["../../models/chemprop-ir/ir_models_data/experiment_model/model_files"]
   }
   ```
3. If needed, modify the path to use absolute paths:
   ```json
   {
       "checkpoint_dir": ["C:/windsurf_repo/MultiModalSpectralTransformer/models/chemprop-ir/ir_models_data/experiment_model/model_files"]
   }
   ```
   Note: Use forward slashes even in Windows for JSON configuration files

4. Save the file

## Step 4: Fix Conda Environment Activation

When activating Conda environments in Windows batch scripts, use environment names instead of full paths.

### Instructions:

1. Locate any batch scripts (`.bat` files) that activate Conda environments
2. Replace any instances of:
   ```batch
   conda activate C:\path\to\environment
   ```
   With:
   ```batch
   call C:\path\to\conda\Scripts\activate.bat
   conda activate environment_name
   ```
3. Save the modified batch files

## Step 5: Test IR Configuration Loading

Verify that the IR configuration can be loaded correctly.

### Instructions:

1. Navigate to the `MMST_website\Runfolder` directory
2. Create a test script named `test_ir_config.py`:
   ```python
   from MMT_import import load_configs
   import os
   
   try:
       print("Current working directory:", os.getcwd())
       IR_config, config = load_configs()
       print("IR config loaded successfully")
       print(f"IR_config checkpoint_dir: {IR_config.checkpoint_dir}")
       
       # Check if checkpoint directory exists
       if os.path.exists(IR_config.checkpoint_dir[0]):
           print(f"Checkpoint directory exists: {IR_config.checkpoint_dir[0]}")
           print("Files in directory:")
           for file in os.listdir(IR_config.checkpoint_dir[0]):
               print(f"  - {file}")
       else:
           print(f"Checkpoint directory does not exist: {IR_config.checkpoint_dir[0]}")
           
       print(f"IR_config test_path: {IR_config.test_path}")
   except Exception as e:
       print(f"Error loading IR config: {str(e)}")
       import traceback
       print(traceback.format_exc())
   ```
3. Run the test script:
   ```
   python test_ir_config.py
   ```
4. Verify that the configuration loads correctly and the checkpoint directory exists

## Step 6: Test IR Simulation

Test the IR simulation functionality with a simple SMILES string.

### Instructions:

1. Create a test script named `test_ir_simulation.py`:
   ```python
   import os
   import sys
   from MMT_import import load_configs, sim_and_display
   
   try:
       # Create a test SMILES file
       test_dir = "test_data"
       os.makedirs(test_dir, exist_ok=True)
       
       test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
       test_file = os.path.join(test_dir, "test_smiles.csv")
       
       with open(test_file, "w") as f:
           f.write("SMILES\n")
           f.write(f"{test_smiles}\n")
       
       print(f"Created test SMILES file at: {os.path.abspath(test_file)}")
       
       # Load configs
       IR_config, config = load_configs()
       
       # Set the SMILES path in config
       config.SGNN_csv_gen_smi = test_file
       config.simulated = True
       
       # Run simulation
       print("Starting simulation...")
       config = sim_and_display()
       print("Simulation successful")
       
       # Check if output files were created
       print("\nChecking output files:")
       expected_ir_folder = config.IR_data_folder_display
       print(f"IR data folder: {expected_ir_folder}")
       
       if os.path.exists(expected_ir_folder):
           print(f"IR data folder exists")
           print("Files in IR data folder:")
           for file in os.listdir(expected_ir_folder):
               print(f"  - {file}")
       else:
           print(f"IR data folder does not exist")
           
   except Exception as e:
       print(f"Simulation failed: {str(e)}")
       import traceback
       print(traceback.format_exc())
   ```
2. Run the test script:
   ```
   python test_ir_simulation.py
   ```
3. Verify that the simulation runs successfully and creates output files

## Step 7: Test Website Integration

Test the IR simulation through the website interface.

### Instructions:

1. Start the Flask application:
   ```
   python app.py
   ```
2. Open a web browser and navigate to the website (usually `http://localhost:5000`)
3. Upload a SMILES file or use the provided examples
4. Click on the "Simulate" button
5. Verify that the IR spectrum is generated and displayed

## Troubleshooting Common Errors

### Error: "IR checkpoint directory not found"

**Solution:**
- Verify that the directory structure is created correctly
- Check that the path in `ir_config_V8.json` matches the actual directory path
- Try using absolute paths instead of relative paths

### Error: "ModuleNotFoundError" when running scripts

**Solution:**
- Ensure the Conda environment is activated correctly
- Use `conda activate <env_name>` instead of full paths
- Verify that all required packages are installed in the environment

### Error: "File not found" for SMILES files

**Solution:**
- Check the path handling in `sim_and_display()` function
- Ensure paths don't have leading slashes when checking file existence
- Use `os.path.normpath()` to normalize paths before checking

### Error: "Invalid column count for IR data"

**Solution:**
- Verify that the IR CSV files have the correct format (single column of absorbance values)
- Check that the IR simulation is generating files correctly

## Additional Debugging Tips

### Add Debug Logging

Add more detailed logging to the simulation process:

```python
# Add to sim_and_display() function
import logging
logging.basicConfig(filename='ir_simulation.log', level=logging.DEBUG)
logging.debug("Starting simulation...")
logging.debug(f"IR_config checkpoint_dir: {IR_config.checkpoint_dir}")
# Add more logging statements as needed
```

### Check File Permissions

Ensure that the application has read/write permissions for all necessary directories:

```python
import os

def check_permissions(path):
    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    print(f"Path: {path}")
    print(f"  Readable: {readable}")
    print(f"  Writable: {writable}")

# Check permissions for important directories
check_permissions("models/chemprop-ir/ir_models_data/experiment_model/model_files/")
check_permissions("MMST_website/Runfolder/")
```

### Test Path Resolution

Test that paths are being resolved correctly:

```python
import os

def test_path_resolution(relative_path):
    script_dir = os.path.dirname(__file__)
    base_path = os.path.abspath(os.path.join(script_dir, ''))
    resolved_path = os.path.join(base_path, relative_path)
    normalized_path = os.path.normpath(resolved_path)
    
    print(f"Relative path: {relative_path}")
    print(f"Script directory: {script_dir}")
    print(f"Base path: {base_path}")
    print(f"Resolved path: {resolved_path}")
    print(f"Normalized path: {normalized_path}")
    print(f"Path exists: {os.path.exists(normalized_path)}")

# Test with IR config path
test_path_resolution("../../models/chemprop-ir/ir_models_data/experiment_model/model_files")
```

## Conclusion

By following these steps, you should be able to fix the IR model integration issues in the MultiModalSpectralTransformer website. The key is to ensure that:

1. The required directory structure exists
2. The IR model files are placed in the correct location
3. Path resolution is handled correctly
4. Conda environments are activated properly

Once these issues are resolved, the IR simulation functionality should work correctly, allowing the website to generate and display IR spectra for the provided SMILES structures.
