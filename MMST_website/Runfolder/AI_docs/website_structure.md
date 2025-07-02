# MultiModalSpectralTransformer Website Documentation

## Repository Structure

```
MultiModalSpectralTransformer/
├── MMST_website/
│   └── Runfolder/
│       ├── AI_docs/                 # Documentation directory
│       ├── CSV_files/               # Test data files
│       │   ├── ML_NMR_1H_combined_ZINC_test_10x100.csv
│       │   ├── ML_NMR_5M_XL_13C_test_10x100.csv
│       │   ├── ML_NMR_5M_XL_COSY_test_10x100.csv
│       │   ├── ML_NMR_5M_XL_HSQC_test_10x100.csv
│       │   └── website_real_example/
│       ├── Log_Folder/              # Logs from website operations
│       ├── Upload_Folder/           # Temporary storage for uploaded files
│       ├── templates/               # HTML templates
│       │   ├── index.html           # Main website interface
│       │   └── upload.html          # File upload interface
│       ├── app.py                   # Main Flask application
│       ├── functions.py             # Helper functions for the website
│       ├── MMT_import.py            # Imports and utilities from the main project
│       ├── config_V8.json           # Main configuration file
│       └── ir_config_V8.json        # IR model configuration
├── models/                          # [MISSING] Expected location for models
│   └── chemprop-ir/
│       └── ir_models_data/
│           └── experiment_model/
│               └── model_files/     # Expected location for IR model files
├── chemprop-ir/                     # IR simulation codebase
├── utils_MMT/                       # Utility functions for the main project
└── [Other project directories]
```

## Key Components

### Configuration Files

1. **config_V8.json**: Main configuration file containing paths to models, datasets, and parameters for the website.
2. **ir_config_V8.json**: Configuration for the IR simulation functionality.

### Python Files

1. **app.py**: Main Flask application that handles routes and serves the website.
2. **functions.py**: Helper functions for data processing and visualization.
3. **MMT_import.py**: Imports from the main project and utilities for model loading.

### HTML Templates

1. **index.html**: Main interface for interacting with the website.
2. **upload.html**: Interface for uploading spectral data files.

## Website Functionality Flow

### Main Page (index.html)

The main page provides several functionalities:
- Molecule visualization
- Spectral data visualization (NMR, IR)
- Model testing and simulation
- Navigation between molecules

### Data Upload (upload.html)

Allows uploading spectral data files:
- 1H NMR
- 13C NMR
- HSQC
- COSY
- IR

### Backend Processing Flow

1. User uploads files or selects existing data
2. Data is processed by functions in `functions.py`
3. Models are loaded using functions in `MMT_import.py`
4. Results are displayed on the main page

## Critical Dependencies

1. **IR Model Files**: Expected at `models/chemprop-ir/ir_models_data/experiment_model/model_files/`
   - **ISSUE**: This directory structure is missing in the repository

2. **Conda Environment**: 
   - When activating Conda environments in Windows batch scripts, using `conda activate <env_name>` is more reliable than using full paths

## Key Routes in app.py

1. **/** - Main page
2. **/upload** - File upload interface
3. **/simulate/<path:SMILES_Path>** - Simulate spectra for given SMILES
4. **/plot_nmr** - Generate and display NMR plots
5. **/molecule_image/<int:index>** - Generate molecule images
6. **/test_model/<path:Checkpoint_Path>/<int:MNS_Value>/<string:spectral_types>** - Test model with parameters

## Data Flow Diagram

```
User Input → Flask Routes → Data Processing → Model Inference → Visualization → User Interface
   ↑                                                                 |
   └─────────────────────── Results Feedback ─────────────────────────┘
```

## Common Issues

1. **Missing IR Model Files**: 
   - The configuration expects IR model files at `models/chemprop-ir/ir_models_data/experiment_model/model_files/`
   - This directory structure needs to be created and populated with model files

2. **Conda Environment Activation**:
   - In Windows batch scripts, use `conda activate <env_name>` instead of full paths
   - This is especially important after initializing the base Conda environment

3. **File Path Issues**:
   - Many paths in the configuration files use relative paths that may not resolve correctly
   - Check that all paths in config_V8.json and ir_config_V8.json point to valid locations
