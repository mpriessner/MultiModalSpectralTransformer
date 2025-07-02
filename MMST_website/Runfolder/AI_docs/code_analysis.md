# MultiModalSpectralTransformer Website Code Analysis

## Core Files and Their Functions

### 1. app.py

The main Flask application that serves as the backbone of the website. Key functionalities include:

- **Route Handling**: Manages URL endpoints and their corresponding functions
- **Data Processing**: Processes user inputs and prepares data for visualization
- **Model Integration**: Interfaces with the MultiModalSpectralTransformer models

Key routes:
- `/`: Main page (index.html)
- `/upload`: File upload interface
- `/simulate/<path:SMILES_Path>`: Simulates spectral data for given SMILES
- `/plot_nmr`: Generates NMR plots
- `/molecule_image/<int:index>`: Generates molecule images
- `/test_model/<path:Checkpoint_Path>/<int:MNS_Value>/<string:spectral_types>`: Tests model with parameters

### 2. functions.py

Contains helper functions for data processing and visualization:

- `write_to_log_file()`: Logs messages to a file and updates the config
- `process_probabilities()`: Processes probability data for molecule visualization
- `generate_colored_molecule()`: Creates colored molecule visualizations
- `parse_NMR_csv()`: Parses NMR data from CSV files
- `parse_SMILES_csv()`: Parses SMILES data from CSV files

### 3. MMT_import.py

Handles imports and utilities from the main project:

- `load_json_dics()`: Loads JSON dictionaries for model operation
- `load_configs()`: Loads configuration files
- `sim_and_display()`: Simulates and prepares data for display

## Component Interactions

### Data Flow

1. **User Input → Flask Routes**:
   - User interacts with the website through forms and buttons
   - Flask routes in app.py capture these interactions

2. **Flask Routes → Data Processing**:
   - Route handlers call functions from functions.py to process data
   - For example, `/simulate` calls `sim_and_display()` from MMT_import.py

3. **Data Processing → Model Inference**:
   - Processed data is passed to models for inference
   - Models are loaded using functions from MMT_import.py

4. **Model Inference → Visualization**:
   - Model outputs are processed for visualization
   - Functions like `generate_colored_molecule()` create visual elements

5. **Visualization → User Interface**:
   - Visual elements are sent to the frontend
   - JavaScript in index.html handles displaying these elements

### HTML-JavaScript-Python Interaction

#### index.html

The main interface contains JavaScript functions that:
- Make AJAX calls to Flask routes
- Update the UI with returned data
- Handle user interactions

Key JavaScript functions:
- `loadMolecule()`: Loads molecule data and image
- `loadSpectrum()`: Loads spectral data for visualization
- `simulateData()`: Triggers data simulation
- `testModel()`: Tests the model with parameters

#### app.py to JavaScript Data Flow

1. JavaScript makes AJAX calls to Flask routes
2. Flask routes process the request and return data (often as JSON)
3. JavaScript updates the UI with the returned data

## Critical Dependencies and Potential Issues

### 1. IR Model Files

The IR simulation functionality depends on model files expected at:
```
models/chemprop-ir/ir_models_data/experiment_model/model_files/
```

**Issue**: This directory structure is missing, which will cause the IR simulation to fail.

**Code Reference**:
```python
# In MMT_import.py
IR_config_dict = load_config(ir_config_path)
IR_config = parse_arguments(IR_config_dict)
modify_predict_args(IR_config)
```

### 2. Path Resolution

Many paths in the configuration files use relative paths that may not resolve correctly.

**Issue**: If the application is run from a different directory, these paths may not resolve correctly.

**Code Reference**:
```python
# In MMT_import.py
def load_configs():
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build paths relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, ''))
    
    ir_config_path = os.path.join(base_path, 'ir_config_V8.json')
    config_path = os.path.join(base_path, 'config_V8.json')
```

### 3. Conda Environment Activation

When activating Conda environments in Windows batch scripts, using `conda activate <env_name>` is more reliable than using full paths.

**Issue**: Using full paths to environments can cause modules not to be found.

## Troubleshooting Guide

### Missing IR Model Files

1. Create the directory structure:
   ```
   models/chemprop-ir/ir_models_data/experiment_model/model_files/
   ```
2. Populate with the necessary model files
3. Update the paths in ir_config_V8.json if needed

### Path Resolution Issues

1. Check all paths in config_V8.json and ir_config_V8.json
2. Ensure they point to valid locations
3. Consider using absolute paths for critical files

### Data Processing Issues

If data processing fails:
1. Check the log files in Log_Folder
2. Verify that input files match the expected format
3. Check that all dependencies are installed

## Website Component Diagram

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|    index.html    |     |     app.py       |     |   MMT_import.py  |
|                  |     |                  |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         | AJAX Calls             | Function Calls         | Imports
         v                        v                        v
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  JavaScript      |     |   functions.py   |     |   utils_MMT      |
|  Functions       |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         |                        | Uses                   | Uses
         v                        v                        v
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  User Interface  |     |   Data Processing|     |   Models         |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
```
