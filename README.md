# MultiModalSpectralTransformer

MultiModalSpectralTransformer is a transformer-based architecture that integrates various spectroscopic modalities (NMR, HSQC, COSY, IR) for automated molecular structure prediction, complete with a data generation pipeline and user-friendly HTML interface.

Implementation of the following publication: 

**Advancing Structure Elucidation with a Flexible Multi-Spectral AI Model**

Publication:
- Preprint: [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/67339b2df9980725cff94c52)
- Data Repository part 1: [Zenodo](https://doi.org/10.5281/zenodo.16076914)
- Data Repository part 2: [Zenodo](https://doi.org/10.5281/zenodo.16257786)
- Data Repository part 3: [Zenodo](https://doi.org/10.5281/zenodo.16283829)
- 
![MultiModalSpectralTransformer Architecture](dump/TOC.png)

## Computational Requirements

This project requires significant computational resources:

- **GPU**: A high-performance GPU is necessary. We recommend using an NVIDIA GPU with CUDA 11.1 support (e.g., NVIDIA V100 or K80).
- **Memory**: At least 16GB RAM to handle datasets and model training.
- **Storage**: At least 50GB storage space for datasets, model checkpoints, and results.
- **Python**: Python 3.7.x is required (tested with Python 3.7.12).

Please ensure your system meets these requirements before proceeding with the installation and usage of MultiModalSpectralTransformer.


## Software Usage Instructions

Detailed instructions on how to use the software, including the full improvement cycle workflow and the HTML GUI interface, are provided in the Electronic Supplementary Information (ESI) of the paper. Please refer to Section 3 of the ESI for a comprehensive user manual.

The ESI contains:
- Step-by-step guide for data preparation
- Instructions for model training and fine-tuning
- Tutorial on using the improvement cycle
- Guide for interpreting model outputs and explanations
- Troubleshooting tips and best practices

## Data and Model Setup

### Required Downloads from Zenodo

Before using this software, you need to download the necessary data and pre-trained models from our Zenodo repository: [https://doi.org/10.5281/zenodo.14712886](https://doi.org/10.5281/zenodo.14712886)

You'll need to download the following files:

1. **Models (Required)**: Download `models.zip` and extract its contents into the `models` directory of this repository
2. **Data (Required)**: Download `data.zip` and extract its contents into the `data` directory of this repository
3. **Experiment Data (Required for Reproducibility)**: Download experiment data and place it in the `experiment` directory to reproduce the figures and results from the paper
4. **Extra Assets (Optional)**: Additional resources for extended functionality

### Folder Structure

After cloning the repository and adding the required data and models, your folder structure should look like this:

```
MultiModalSpectralTransformer
│
├── 📁 AI_docs                # Documentation related to the AI model
├── 📁 chemprop-IR            # IR spectrum prediction model
├── 📁 data                   # All datasets (download from Zenodo)
│   ├── 📁 IBM_dataset        
│   ├── 📁 PubChem_dataset
│   ├── 📁 test_data
│   ├── 📁 ZINC_4000
│   └── 📁 ZINC_dataset
├── 📁 deep-molecular-optimization # Molecule optimization code
├── 📁 dump                   # Temporary files and outputs
├── 📁 experiments            # General experiment outputs
├── 📁 HSQC_results           # Results specific to HSQC experiments
├── 📁 MMT_website            # Web interface files
├── 📁 models                 # Pre-trained models (download from Zenodo)
│   ├── 📁 chemprop-ir
│   ├── 📁 mmst
│   ├── 📁 mol2mol
│   ├── 📁 sgnn
├── 📁 nmr_sgnn_norm
├── 📁 past_experiments / ChemXriv # Notebooks and data to reproduce paper experiments
│   ├── 📁 0.0_Experiment_Training_Strategy
│   ├── 📁 1.0_Experiment_Trainings_Experiments
│   └── 📁 ... (and other experiment folders)
├── 📁 scripts                # Utility scripts
├── 📁 utils_MMT              # Core utilities and functions
└── [Various notebooks]       # Jupyter notebooks for different tasks
```

Ensure all downloaded files are placed in their respective directories as shown above.


## Notebooks

The project includes several Jupyter notebooks for different purposes:

1. **1.0_Experiment_Notebook.ipynb**
   - This notebook is used to reproduce the experiments.
   
2. **2.0_NMR_Data_Generation.ipynb**
   - This notebook is used to generate simulated NMR data using the SGNN network.
   - [Link to SGNN paper](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp04542g#:~:text=Abstract,limited%20to%20relatively%20small%20molecules.)

3. **3.0_Chemprop_IR_Data_Generation.ipynb**
   - This notebook is used to produce simulated IR data using the Chemprop-IR network.
   - [Link to Chemprop-IR paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00055)

4. **4.0_Explainability_plot.ipynb**
   - Visualizes molecules with color-coded atoms showing probabilities
   - Creates SVGs, labels, and SMILES string visualizations
   - Supports colored and non-colored molecule rendering from pickle files

Please refer to these notebooks for detailed procedures on data generation and experiment reproduction.


## Installation

### Prerequisites

- **Conda**: Ensure you have Conda installed on your system.
- **CUDA**: CUDA 11.1 is required for the GPU acceleration (required for PyTorch).

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/MultiModalSpectralTransformer.git
   cd MultiModalSpectralTransformer
   ```

2. **Download Required Data and Models**:
3. **Download the necessary files from our Zenodo repository**:
  [https://doi.org/10.5281/zenodo.14712886](https://doi.org/10.5281/zenodo.16076914)
  [https://doi.org/10.5281/zenodo.14712886](https://doi.org/10.5281/zenodo.16257786)
   
   ```bash
   # Create directories if they don't exist
   mkdir -p data models experiment
   
   # Download and extract model files (replace with actual download links from Zenodo)
   wget -O models.zip <zenodo-models-download-url>
   unzip models.zip -d models/
   
   # Download and extract data files
   wget -O data.zip <zenodo-data-download-url>
   unzip data.zip -d data/
   
   # Download and extract experiment files for reproducibility
   wget -O experiment.zip <zenodo-experiment-download-url>
   unzip experiment.zip -d experiment/
   ```

4. **Environment Setup**:
   We provide an installation script that sets up all the necessary dependencies:
   
   ```bash
   # Create a new conda environment
   conda create -y -c conda-forge -n NMR_Structure_Elucidator python=3.7.12
   
   # Activate the environment
   conda activate NMR_Structure_Elucidator
   
   # Run the installation script
   bash installs.sh
   ```
   
   The `installs.sh` script will install all required packages including PyTorch, RDKit, and other dependencies needed for the project.

5. **Verify Installation**:
   Test that the environment is properly set up by running one of the simpler notebooks:
   
   ```bash
   jupyter lab
   ```
   
   Then open one of the notebooks to verify functionality.

## Usage

### Running the Web Interface

1. **Start the Web Application**:
   ```bash
   cd MMT_website
   python app.py
   ```

2. **Access the Application**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

### Using the Jupyter Notebooks

We provide several Jupyter notebooks for different aspects of the workflow:

1. **Run Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Select a Notebook**:
   - `1.0_Experiment_Notebook.ipynb` - For running experiments and evaluating results
   - `2.0_Automatic_NMR_Data_Generation.ipynb` - For generating synthetic NMR data
   - `3.0_Chemprop_IR_Data_Generation_.ipynb` - For generating IR spectral data
   - `4.0_Explainability_plot.ipynb` - For visualizing model interpretations

### Common Issues

1. **CUDA/GPU Issues**:
   - Error: `CUDA error: no kernel image is available for execution on the device`
   - Solution: Ensure your CUDA version matches the requirements (CUDA 11.1) and that your GPU drivers are up to date.

2. **Missing Data Files**:
   - Error: `FileNotFoundError: [Errno 2] No such file or directory: 'data/...'`
   - Solution: Make sure you've downloaded and extracted all required data files from Zenodo to the correct locations.

3. **Python Environment Issues**:
   - Error: `ModuleNotFoundError: No module named 'xxx'`
   - Solution: Double-check that all dependencies are installed in your conda environment. Run `pip list` to verify.

4. **Model Loading Errors**:
   - Error: `torch.nn.modules.module.ModuleAttributeError: 'xxx' object has no attribute 'xxx'`
   - Solution: Ensure you're using the correct model files from Zenodo and that they match the expected format.

5. **Accessing Web Interface on Remote Servers**:
   - Issue: When running the application on a remote or virtual node, you can't access the web interface via browser.
   - Solution: Set up SSH port forwarding to tunnel the application port from the remote server to your local machine:
     ```bash
     ssh -L 5000:localhost:5000 <username>@<node>
     ```
     This command forwards port 5000 on the remote server to port 5000 on your local machine, allowing you to access the web interface by visiting `http://localhost:5000` in your browser.
   - Note: Make sure your SSH connection remains active while using the web interface. For persistent connections, consider using tools like `tmux` or `screen` on the remote server.
  
If you encounter persistent issues, please check the GitHub repository for updated troubleshooting guidance or open an issue with detailed information about your problem.

## Contributing

We welcome contributions to improve the MultiModalSpectralTransformer. Please fork the repository and submit a pull request.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License. For more details, see the LICENSE file.
