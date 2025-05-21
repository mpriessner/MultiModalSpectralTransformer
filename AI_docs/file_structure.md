# Git Repository Structure and Files

This document outlines the folder structure and key files within this Git repository.

## Full File Breakdown
```
text
.
├── 1.8_Experiment_Notebook_.ipynb
├── 2.0_Automatic_NMR_Data_Generation.ipynb
├── 3.0_Chemprop_IR_Data_Generation_.ipynb
├── 4.0_Explainability_plot.ipynb
├── LICENCE
├── README.md
├── installs.sh
├── itos.json
├── itos_MF.json
├── stoi.json
├── stoi_MF.json
├── .vscode
│   └── settings.json
├── dump
│   ├── TOC.png
│   └── __init__.py
├── scripts
│   ├── PC_0_250.txt
│   ├── PC_0_250_new.sh
│   ├── PC_0_250_v2.txt
│   ├── PC_0_250_v3.txt
│   ├── PC_250_350.sh
│   ├── PC_250_350.txt
│   ├── PC_350_500.sh
│   ├── PC_350_500.txt
│   ├── Untitled.ipynb
│   ├── ZINC_0_250.sh
│   ├── ZINC_0_250.txt
│   ├── ZINC_250_350_4000.sh
│   ├── ZINC_250_350_4000_v1.txt
│   ├── script_PC_0_250.py
│   ├── script_PC_250_350.py
│   ├── script_PC_350_500.py
│   ├── script_ZINC_250_350.py
│   ├── script_ZINC_250_350_4000.py
│   └── test.ipynb
└── utils_MMT
    ├── MT_functions_v15_4.py
    ├── Qformer_v15_4.py
    ├── blip_functions_v15_4.py
    ├── clip_functions_v15_4.py
    ├── clustering_visualization_v15_4.py
    ├── config_V8.json
    ├── cosy_nmr_reconstruction_v15_4.py
    ├── data_generation_v15_4.py
    ├── dataloaders_pl_v15_4.py
    ├── execution_function_v15_4.py
    ├── experiment_function_v15_4.py
    ├── functions_HSQC_sim_v15_4.py
    ├── helper_functions_pl_v15_4.py
    ├── hsqc_nmr_reconstruction_v15_4.py
    ├── improvement_cycle_neg_examples_v15_4.py
    ├── ir_config_V8.json
    ├── ir_simulation_v15_4.py
    ├── mmt_result_test_functions_15_4.py
    ├── models_BLIP_v15_4.py
    ├── models_CLIP_v15_4.py
    ├── models_MMT_v15_4.py
    ├── molformer_functions_v15_4.py
    ├── nmr_calculation_from_dft_v15_4.py
    ├── plotting_v15_4.py
    ├── run_batch_gen_val_MMT_v15_4.py
    ├── sgnn_code_pl_v15_4.py
    ├── similarity_functions_v15_4.py
    ├── smi_augmenter_v15_4.py
    ├── train_test_functions_pl_v15_4.py
    └── validate_generate_MMT_v15_4.
```
## File Breakdown with Descriptions

### ChemPropIR

*   `3.0_Chemprop_IR_Data_Generation_.ipynb`: Jupyter notebook for generating data using Chemprop for Infrared (IR) spectroscopy.
*   `ir_config_V8.json`: Configuration file for IR experiments (likely used by Chemprop or related scripts).
*   `ir_simulation_v15_4.py`: Python script for simulating IR spectra.

### Deep Molecular Optimization (Mole2Mole)

*   `1.8_Experiment_Notebook_.ipynb`: Experiment notebook, potentially for training or evaluating Mole2Mole models.
*   `itos.json`: Integer-to-string mapping, likely for tokenizing molecular representations (SMILES).
*   `itos_MF.json`: Integer-to-string mapping for molecular fragments or features.
*   `stoi.json`: String-to-integer mapping, reverse of `itos.json`.
*   `stoi_MF.json`: String-to-integer mapping for molecular fragments or features, reverse of `itos_MF.json`.
*   `utils_MMT/molformer_functions_v15_4.py`: Python script containing functions related to the Molformer model, a transformer-based model for molecules.
*   `utils_MMT/smi_augmenter_v15_4.py`: Python script for augmenting SMILES strings (molecular representations).

### SGNN

*   `utils_MMT/sgnn_code_pl_v15_4.py`: Python script containing code for a Structure-Guided Neural Network (SGNN).

### MMT (Multimodal Molecular Transformer)

*   `2.0_Automatic_NMR_Data_Generation.ipynb`: Jupyter notebook for automatic generation of NMR data.
*   `4.0_Explainability_plot.ipynb`: Jupyter notebook for generating explainability plots for models.
*   `utils_MMT/MT_functions_v15_4.py`: Python script containing functions for the Multimodal Transformer (MT) model.
*   `utils_MMT/Qformer_v15_4.py`: Python script related to the Q-Former model, likely used in a multimodal context.
*   `utils_MMT/blip_functions_v15_4.py`: Python script with functions for a BLIP-like model, possibly for image-text or multimodal tasks.
*   `utils_MMT/clip_functions_v15_4.py`: Python script with functions for a CLIP-like model, possibly for image-text or multimodal tasks.
*   `utils_MMT/clustering_visualization_v15_4.py`: Python script for clustering and visualizing results.
*   `utils_MMT/config_V8.json`: Configuration file for MMT experiments.
*   `utils_MMT/cosy_nmr_reconstruction_v15_4.py`: Python script for reconstructing COSY NMR spectra.
*   `utils_MMT/data_generation_v15_4.py`: Python script for generating data for MMT models.
*   `utils_MMT/dataloaders_pl_v15_4.py`: Python script containing PyTorch Lightning DataLoaders for MMT.
*   `utils_MMT/execution_function_v15_4.py`: Python script for executing MMT training or evaluation runs.
*   `utils_MMT/experiment_function_v15_4.py`: Python script with functions for running MMT experiments.
*   `utils_MMT/functions_HSQC_sim_v15_4.py`: Python script containing functions for simulating HSQC NMR spectra.
*   `utils_MMT/helper_functions_pl_v15_4.py`: Python script with helper functions for PyTorch Lightning.
*   `utils_MMT/hsqc_nmr_reconstruction_v15_4.py`: Python script for reconstructing HSQC NMR spectra.
*   `utils_MMT/improvement_cycle_neg_examples_v15_4.py`: Python script for an improvement cycle potentially involving negative examples.
*   `utils_MMT/mmt_result_test_functions_15_4.py`: Python script with test functions for MMT results.
*   `utils_MMT/models_BLIP_v15_4.py`: Python script defining BLIP-like models for MMT.
*   `utils_MMT/models_CLIP_v15_4.py`: Python script defining CLIP-like models for MMT.
*   `utils_MMT/models_MMT_v15_4.py`: Python script defining the core MMT models.
*   `utils_MMT/nmr_calculation_from_dft_v15_4.py`: Python script for calculating NMR properties from DFT outputs.
*   `utils_MMT/plotting_v15_4.py`: Python script for generating plots related to MMT.
*   `utils_MMT/run_batch_gen_val_MMT_v15_4.py`: Python script for running batch generation and validation for MMT.
*   `utils_MMT/similarity_functions_v15_4.py`: Python script with functions for calculating similarity.
*   `utils_MMT/train_test_functions_pl_v15_4.py`: Python script with functions for training and testing MMT models using PyTorch Lightning.
*   `utils_MMT/validate_generate_MMT_v15_4.py`: Python script for validating and generating data with MMT models.

### Other Files

*   `LICENCE`: The license file for the repository.
*   `README.md`: The main README file providing an overview of the project.
*   `installs.sh`: Shell script for installing necessary dependencies.
*   `.vscode/settings.json`: VS Code settings file.
*   `dump/TOC.png`: Table of Contents image (likely for a document or presentation).
*   `dump/__init__.py`: Initialization file for the `dump` directory.
*   `scripts/`: Directory containing various shell scripts and Python scripts.
    *   `PC_0_250.txt`, `PC_0_250_new.sh`, `PC_0_250_v2.txt`, `PC_0_250_v3.txt`: Files related to processing data chunks (0-250).
    *   `PC_250_350.sh`, `PC_250_350.txt`: Files related to processing data chunks (250-350).
    *   `PC_350_500.sh`, `PC_350_500.txt`: Files related to processing data chunks (350-500).
    *   `Untitled.ipynb`: An untitled Jupyter notebook, possibly for testing or exploration.
    *   `ZINC_0_250.sh`, `ZINC_0_250.txt`: Files related to processing ZINC database data chunks (0-250).
    *   `ZINC_250_350_4000.sh`, `ZINC_250_350_4000_v1.txt`: Files related to processing ZINC database data chunks (250-350), potentially up to 4000 entries.
    *   `script_PC_0_250.py`, `script_PC_250_350.py`, `script_PC_350_500.py`: Python scripts for processing data chunks.
    *   `script_ZINC_250_350.py`, `script_ZINC_250_350_4000.py`: Python scripts for processing ZINC data chunks.
    *   `test.ipynb`: A test Jupyter notebook.