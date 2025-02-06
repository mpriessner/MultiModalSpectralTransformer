# MultiModalSpectralTransformer

MultiModalSpectralTransformer is a transformer-based architecture that integrates various spectroscopic modalities (NMR, HSQC, COSY, IR) for automated molecular structure prediction, complete with a data generation pipeline and user-friendly HTML interface.

Implementation of the following publication: 

**Advancing Structure Elucidation with a Flexible Multi-Spectral AI Model**

Publication:
- Preprint: [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/67339b2df9980725cff94c52)
- Data Repository: [Zenodo](https://doi.org/10.5281/zenodo.14712886)

![MultiModalSpectralTransformer Architecture](dump/MMT.png)

## Computational Requirements

This project requires significant computational resources:

- **GPU**: A high-performance GPU is necessary. We recommend using an NVIDIA V100 or K80 GPU.
- **Memory**: Sufficient RAM to handle large datasets and model training.
- **Storage**: Adequate storage space for datasets, model checkpoints, and results.

Please ensure your system meets these requirements before proceeding with the installation and usage of MultiModalTransformer.

## Folder Structure

After cloning the repository and adding the extra models, your folder structure should look like this:
```
MultiModalTransformer
│
├── 📁 chemprop-IR
├── 📁 deep-molecular-optimization
├── 📁 dump
├── 📁 MMT_website
├── 📁 models
├── 📁 nmr_sgnn_norm
└── 📁 utils_MMT
```

Ensure that after extracting the extra libraries, your folder structure matches this layout.


## Notebooks

The project includes several Jupyter notebooks for different purposes:

1. **1.0_Experiment_Notebook.ipynb**
   - This notebook is used to reproduce the experiments.
   - Note: The paths for the pkl files need to be changed according to the extraction folder where they are stored.
   - [Link to related paper]
   
2. **2.0_NMR_Data_Generation.ipynb**
   - This notebook is used to generate simulated NMR data using the SGNN network.
   - [Link to SGNN paper](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp04542g#:~:text=Abstract,limited%20to%20relatively%20small%20molecules.)

3. **3.0_Chemprop_IR_Data_Generation.ipynb**
   - This notebook is used to produce simulated IR data using the Chemprop-IR network.
   - [Link to Chemprop-IR paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00055)

4. **4.0_Chemprop_IR_Data_Generation.ipynb**
   - Visualizes molecules with color-coded atoms showing probabilities
   - Creates SVGs, labels, and SMILES string visualizations
   - Supports colored and non-colored molecule rendering from pickle files

Please refer to these notebooks for detailed procedures on data generation and experiment reproduction.


## Installation

### Prerequisites

- **Conda**: Ensure you have Conda installed on your system.

### Setup

1. **Create Conda Environment**:
   ```
   conda create -y -c conda-forge -n NMR_Structure_Elucidator python=3.7.6
   ```

2. **Activate the Environment**:
   ```
   conda activate NMR_Structure_Elucidator
   ```

3. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd <path-of-the-cloned-github-code-folder>/MultiModalTransformer
   ```

4. **Download and Extract Extra Libraries**:
   - Download the `extra_libraries.zip` file from the following Zenodo link:
     [Zenodo Link for extra_libraries.zip]

   - Extract the contents of the zip file into the `/MultiModalTransformer` folder of your cloned repository:
     ```
     unzip path/to/extra_libraries.zip -d /path/to/MultiModalTransformer/
     ```
   
   Make sure all extracted files and folders are directly under the `/MultiModalTransformer` directory.

5. **Install Dependencies**:
   ```
   bash installs.sh
   ```

## Usage

1. **Launch the Web Application**:
   ```bash
   python app.py
   ```

2. **Access the Application**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Contributing

We welcome contributions to improve the MultiModalTransformer. Please fork the repository and submit a pull request.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License. For more details, see the LICENSE file.
