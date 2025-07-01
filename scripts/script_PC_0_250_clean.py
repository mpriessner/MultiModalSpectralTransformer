#!/usr/bin/env python3
"""
Cleaned configuration script for MultiModalSpectralTransformer
Path: /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalSpectralTransformer_cleaned/scripts
"""

import matplotlib.pyplot as plt
import json
import os
import random
import glob
import pickle
import sys
from datetime import datetime
import tempfile
import copy
from collections import defaultdict
from argparse import Namespace

# Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm

# Machine learning and data visualization
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PyTorch for deep learning
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# RDKit for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler

# Miscellaneous
from IPython.display import HTML, SVG

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from scripts/

# Add utils to path
sys.path.append(os.path.join(project_root, 'utils_MMT'))

# Import custom modules
try:
    import utils_MMT.clip_functions_v15_4 as cf
    import utils_MMT.MT_functions_v15_4 as mtf
    import utils_MMT.validate_generate_MMT_v15_4 as vgmmt
    import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
    import utils_MMT.clustering_visualization_v15_4 as cv
    import utils_MMT.plotting_v15_4 as pt
    import utils_MMT.execution_function_v15_4 as ex
    import utils_MMT.train_test_functions_pl_v15_4 as ttf
    import utils_MMT.ir_simulation_v15_4 as irs
    import utils_MMT.helper_functions_pl_v15_4 as hf
    import utils_MMT.mmt_result_test_functions_15_4 as mrtf
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure the utils_MMT directory exists in the project root")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_json_dics():
    """Load vocabulary dictionaries"""
    try:
        with open(os.path.join(project_root, 'itos.json'), 'r') as f:
            itos = json.load(f)
        with open(os.path.join(project_root, 'stoi.json'), 'r') as f:
            stoi = json.load(f)
        with open(os.path.join(project_root, 'stoi_MF.json'), 'r') as f:
            stoi_MF = json.load(f)
        with open(os.path.join(project_root, 'itos_MF.json'), 'r') as f:
            itos_MF = json.load(f)
        return itos, stoi, stoi_MF, itos_MF
    except FileNotFoundError as e:
        print(f"Error loading vocabulary files: {e}")
        print("Make sure the vocabulary JSON files exist in the project root")
        raise

# Load vocabularies
itos, stoi, stoi_MF, itos_MF = load_json_dics()
rand_num = str(random.randint(1, 10000000))

# IR Configuration
IR_config_dict = {
    "gpu": list(range(torch.cuda.device_count())),
    "test_path": [os.path.join(project_root, "models/chemprop-ir/ir_models_data/solvation_example/solvation_spectra.csv")],
    "use_compound_names": [False],
    "preds_path": [os.path.join(project_root, "models/chemprop-ir/ir_models_data/ir_preds_test_2.csv")],
    "checkpoint_dir": [os.path.join(project_root, "models/chemprop-ir/ir_models_data/experiment_model/model_files")],
    "spectra_type": ["experimental"],
    "spectra_type_nr": [0],
    "checkpoint_path": [None],
    "batch_size": [50],
    "no_cuda": [False],
    "features_generator": [None],
    "features_path": [None],
    "max_data_size": [100],
    "ensemble_variance": [False],
    "ensemble_variance_conv": [0.0],
}

# Main hyperparameters configuration
hyperparameters = {
    # General project information
    "project": ["MMST_V1"],
    "ran_num": [rand_num],
    "device": ["cuda"],
    "gpu_num": [1],
    "num_workers": [4],
    "data_type": ["sgnn"],
    "execution_type": ["validate_MMT"],
    "syn_data_simulated": [False],
    "training_type": ["clip"],

    # Encoding dicts
    "itos_path": [os.path.join(project_root, "itos.json")],
    "stoi_path": [os.path.join(project_root, "stoi.json")],
    "itos_MF_path": [os.path.join(project_root, "itos_MF.json")],
    "stoi_MF_path": [os.path.join(project_root, "stoi_MF.json")],
    
    # Data settings
    "input_dim_1H": [2],
    "input_dim_13C": [1],
    "input_dim_HSQC": [2],
    "input_dim_COSY": [2],
    "input_dim_IR": [1000],
    "MF_vocab_size": [len(stoi_MF)],
    "MS_vocab_size": [len(stoi)],
    "tr_te_split": [0.9],
    "padding_points_number": [64],
    "data_size": [1000],
    "test_size": [10],
    "model_save_dir": [os.path.join(project_root, "experiments/exp_1/model_save_dir")],
    "ML_dump_folder": [os.path.join(project_root, "experiments/exp_1/dump")],
    "model_save_interval": [10000],
    
    # SGNN data paths
    "use_real_data": [False],
    "ref_data_type": ["1H"],
    "csv_train_path": [os.path.join(project_root, "data/ZINK_dataset/ML_NMR_5M_XL_1H_comb_train_V8.csv")],
    "csv_1H_path_SGNN": [os.path.join(project_root, "data/ZINK_dataset/ML_NMR_5M_XL_1H_comb_train_V8.csv")],
    "csv_13C_path_SGNN": [os.path.join(project_root, "data/ZINK_dataset/ML_NMR_5M_XL_13C_train_V8.csv")],
    "csv_HSQC_path_SGNN": [os.path.join(project_root, "data/ZINK_dataset/ML_NMR_5M_XL_HSQC_train_V8.csv")],
    "csv_COSY_path_SGNN": [os.path.join(project_root, "data/ZINK_dataset/ML_NMR_5M_XL_COSY_train_V8.csv")],
    "csv_IR_MF_path": [''],
    "csv_path_val": [os.path.join(project_root, "data/ZINK_dataset/ML_NMR_5M_XL_1H_comb_test_V8.csv")],
    "IR_data_folder": [os.path.join(project_root, "data/ZINK_dataset/IR_spectra_NN")],
    "pickle_file_path": [""],
    "dl_mode": ['val'],
    "isomericSmiles": [False],
    
    # Transformer Settings
    "training_mode": ["1H_13C_HSQC_COSY_IR_MF_MW"],
    "blank_percentage": [0.0],
    "batch_size": [64],
    "num_epochs": [10],
    "lr_pretraining": [1e-4],
    "lr_finetuning": [5e-5],
    "load_model": [True],
    "checkpoint_path": [os.path.join(project_root, "models/mmst/base_models/1_0_V8i_MMTi_RAW_MW_DROP_Loss_0.112.ckpt")],
    "save_model": [True],
    
    # Model architecture
    "in_size": [len(stoi)],
    "hidden_size": [128],
    "out_size": [len(stoi)],
    "num_encoder_layers": [6],
    "num_decoder_layers": [6],
    "num_heads": [16],
    "forward_expansion": [4],
    "max_len": [128],
    "drop_out": [0.1],
    "fingerprint_size": [512],
    "gen_SMI_sequence": [True],
    "sampling_method": ["mix"],
    "training_setup": ["pretraining"],
    "smi_randomizer": [False],
    
    # SGNN Feedback
    "sgnn_feedback": [False],
    "matching": ["HungDist"],
    "padding": ["NN"],
    "train_weight_min": [None],
    "train_weight_max": [None],
    
    # Training Loss Weighting options
    "weight_validity": [0.0],
    "weight_SMI": [1.0],
    "weight_FP": [0.0],
    "weight_MW": [0],
    "weight_sgnn": [0.0],
    "weight_tanimoto": [0.0],
    "change_loss_weights": [False],
    "increment": [0.01],
    "batch_frequency": [10000],
    
    # Validation
    "beam_size": [1],
    "multinom_runs": [1],
    "temperature": [1],
    "gen_len": [64],
    "pkl_save_folder": [os.path.join(project_root, "experiments/exp_1/pkl_save_folder")],
    
    # Molformer options
    "MF_max_trails": [500],
    "MF_tanimoto_filter": [0.1],
    "MF_filter_higher": [1],
    "MF_delta_weight": [5],
    "MF_generations": [30],
    "MF_model_path": [os.path.join(project_root, "models/mol2mol/Alessandro_big/weights_pubchem_with_counts_and_rank_sanitized.ckpt")],
    "MF_vocab": [os.path.join(project_root, "models/mol2mol/Alessandro_big/vocab_new.pkl")],
    "MF_csv_source_folder_location": [os.path.join(project_root, "deep-molecular-optimization/data/MMP")],
    "MF_csv_source_file_name": ["test_selection_2"],
    "MF_methods": ["MMP"],
    "max_scaffold_generations": [10],
    
    # MMT batch generation
    "MMT_batch": [32],
    "MMT_generations": [4],
    "n_samples": [10],
    "gen_mol_csv_folder_path": [os.path.join(project_root, "data/SGNN_gen_folder")],
    
    # Fine-tuning improvement options
    "train_data_blend": [0],
    "train_data_blend_CLIP": [1000],
    
    # Data generation SGNN
    "SGNN_gen_folder_path": [os.path.join(project_root, "experiments/exp_1/SGNN_gen_folder")],
    "SGNN_csv_gen_smi": [os.path.join(project_root, "data/test_data/IBM_SMI_data_top20.csv")],
    "SGNN_size_filter": [550],
    "SGNN_csv_save_folder": [os.path.join(project_root, "experiments/exp_1/SGNN_gen_folder")],
    "IR_save_folder": [os.path.join(project_root, "experiments/exp_1/IR_data")],

    # CLIP Model settings (Legacy)
    "model_version": [os.path.join(project_root, "models/mmst/OLD/Chemberta_source")],
    "CB_model_path": [os.path.join(project_root, "models/mmst/OLD/Large_300_15.pth")],
    "num_class": [1024],
    "num_linear_layers": [0],
    "use_dropout": [True],
    "use_relu": [False],
    "loss_fn": ["BCEWithLogitsLoss"],
    "CB_embedding": [1024],
    "fp_dim_reduction": [False],
    "pca_components": [300],
    
    # Multimodal Transformer
    "MT_model_path": [os.path.join(project_root, "models/mmst/OLD/MultimodalTransformer_time_1706856620.3718672_Loss_0.202.pth")],
    "MT_embedding": [512],
    
    # Projection Head
    "projection_dim": [512],
    "dropout": [0.1],
    
    # CLIP Training parameters
    "similarity_threshold": [0.6],
    "max_search_size": [10000],
    "weight_delta": [50],
    "CLIP_batch_size": [128],
    "CLIP_NUM_EPOCHS": [10],
    "CLIP_temperature": [1],
    "MT_projection_lr": [1e-3],
    "CB_lr": [1e-4],
    "MT_lr": [1e-5],
    "weight_decay": [1e-3],
    "patience": [1],
    "factor": [0.8],
    "CLIP_continue_training": [True],
    "CLIP_model_path": [os.path.join(project_root, "models/mmst/OLD/MultimodalCLIP_Epoch_9_Loss0.096.ckpt")],
    "CLIP_model_save_dir": [os.path.join(project_root, "models/mmst/OLD/test_CLIP")],
}

def save_config(config, path):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path):
    """Load configuration from JSON file"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {path}")
        return None

def parse_arguments(hyperparameters):
    """Convert hyperparameters dict to Namespace object"""
    parsed_args = {key: val[0] for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)

def setup_directories():
    """Create necessary directories"""
    directories = [
        os.path.join(project_root, "experiments/exp_1/model_save_dir"),
        os.path.join(project_root, "experiments/exp_1/dump"),
        os.path.join(project_root, "experiments/exp_1/pkl_save_folder"),
        os.path.join(project_root, "experiments/exp_1/SGNN_gen_folder"),
        os.path.join(project_root, "experiments/exp_1/IR_data"),
        os.path.join(project_root, "utils_MMT"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def initialize_configs():
    """Initialize and save configuration files"""
    # Setup directories
    setup_directories()
    
    # Parse configurations
    config = parse_arguments(hyperparameters)
    
    # Save IR config
    ir_config_path = os.path.join(project_root, 'utils_MMT/ir_config_V8.json')
    save_config(IR_config_dict, ir_config_path)
    IR_config_dict_loaded = load_config(ir_config_path)
    IR_config = parse_arguments(IR_config_dict_loaded)
    
    # Modify IR config if irs module is available
    try:
        irs.modify_predict_args(IR_config)
    except NameError:
        print("Warning: irs module not available, skipping IR config modification")
    
    # Save main config
    config_path = os.path.join(project_root, 'utils_MMT/config_V8.json')
    save_config(hyperparameters, config_path)
    config_dict = load_config(config_path)
    config = parse_arguments(config_dict)
    
    return config, IR_config

# Analysis and processing functions
def process_pkl_files(folder_path, file_type, ranking_method):
    """Process pickle files in a folder"""
    pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.endswith('.pkl') and file_type in f]
    
    all_rankings = defaultdict(list)
    
    for file_path in pkl_files:
        try:
            file_data = load_data(file_path)
            ranked_molecules = rank_molecules_in_file(file_data, ranking_method)
            
            for molecule in ranked_molecules:
                trg_smi = molecule[0]
                all_rankings[trg_smi].append(molecule)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_rankings

def split_dataset(config, chunk_size: int) -> list:
    """Split dataset into chunks"""
    df = pd.read_csv(config.SGNN_csv_gen_smi)
    return [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def create_chunk_folder(config, idx: int) -> str:
    """Create folder for processing chunks"""
    base_dir = config.model_save_dir
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_folder_name = f"chunk_{idx:03d}_{current_datetime}"
    chunk_folder_path = os.path.join(base_dir, chunk_folder_name)
    
    os.makedirs(chunk_folder_path, exist_ok=True)
    print(f"Created folder for chunk {idx}: {chunk_folder_path}")
    
    return chunk_folder_path

def prepare_data(config, chunk: pd.DataFrame):
    """Prepare data for processing"""
    chunk_csv_path = os.path.join(config.pkl_save_folder, "SGNN_csv_gen_smi.csv")
    chunk.to_csv(chunk_csv_path, index=False)
    config.SGNN_csv_gen_smi = chunk_csv_path 
    config.data_size = len(chunk)
    return config

def generate_simulated_data(config, IR_config):
    """Generate simulated data"""
    config.execution_type = "data_generation"
    if config.execution_type == "data_generation":
        print("\033[1m\033[31mThis is: data_generation\033[0m")
        try:
            config = ex.gen_sim_aug_data(config, IR_config)
            backup_config_paths(config)
        except Exception as e:
            print(f"Error in data generation: {e}")
    return config

def backup_config_paths(config) -> None:
    """Backup configuration paths"""
    config.csv_1H_path_SGNN_backup = copy.deepcopy(config.csv_1H_path_SGNN)
    config.csv_13C_path_SGNN_backup = copy.deepcopy(config.csv_13C_path_SGNN)
    config.csv_HSQC_path_SGNN_backup = copy.deepcopy(config.csv_HSQC_path_SGNN)
    config.csv_COSY_path_SGNN_backup = copy.deepcopy(config.csv_COSY_path_SGNN)
    config.IR_data_folder_backup = copy.deepcopy(config.IR_data_folder)

def load_model_and_data(config, stoi: dict, stoi_MF: dict):
    """Load model and data loaders"""
    try:
        val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
        val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
        model_MMT = mrtf.load_MMT_model(config)
        return model_MMT, val_dataloader, val_dataloader_multi
    except Exception as e:
        print(f"Error loading model and data: {e}")
        raise

def test_model_performance(config, model_MMT, val_dataloader, val_dataloader_multi, 
                          stoi: dict, itos: dict, stoi_MF: dict, itos_MF: dict) -> dict:
    """Test model performance"""
    print("\033[1m\033[31mThis is: test_performance\033[0m")
    
    MW_filter = True
    greedy_full = False
    
    try:
        model_CLIP = mrtf.load_CLIP_model(config)
        
        results = {}
        
        results['results_dict_bl_ZINC_'] = mrtf.run_test_mns_performance_CLIP_3(
            config, model_MMT, model_CLIP, val_dataloader, stoi, itos, MW_filter)
        results['results_dict_bl_ZINC_'], counter = mrtf.filter_invalid_inputs(results['results_dict_bl_ZINC_'])

        results['avg_tani_bl_ZINC_'], html_plot = rbgvm.plot_hist_of_results(results['results_dict_bl_ZINC_'])

        if greedy_full:
            results['results_dict_greedy_bl_ZINC_'], results['failed_bl_ZINC_'] = mrtf.run_test_performance_CLIP_greedy_3(
                config, stoi, stoi_MF, itos, itos_MF)
            results['avg_tani_greedy_bl_ZINC_'], html_plot_greedy = rbgvm.plot_hist_of_results_greedy(
                results['results_dict_greedy_bl_ZINC_'])
        else:
            config, results['results_dict_ZINC_greedy_bl_'] = mrtf.run_greedy_sampling(
                config, model_MMT, val_dataloader_multi, itos, stoi)
            results['avg_tani_greedy_bl_ZINC_'] = results['results_dict_ZINC_greedy_bl_']["tanimoto_mean"]

        results['total_results_bl_ZINC_'] = mrtf.run_test_performance_CLIP_3(
            config, model_MMT, val_dataloader, stoi)
        results['corr_sampleing_prob_bl_ZINC_'] = results['total_results_bl_ZINC_']["statistics_multiplication_avg"][0]

        print("avg_tani, avg_tani_greedy, corr_sampleing_prob")
        print(results['avg_tani_bl_ZINC_'], results['avg_tani_greedy_bl_ZINC_'], results['corr_sampleing_prob_bl_ZINC_'])
        print("Greedy tanimoto results")
        rbgvm.plot_hist_of_results_greedy_new(results['results_dict_ZINC_greedy_bl_'])

        return results
    except Exception as e:
        print(f"Error in model performance testing: {e}")
        raise

def save_results_before(results: dict, config, idx: int) -> None:
    """Save results before processing"""
    variables_to_save = {
        'avg_tani_bl_ZINC': results['avg_tani_bl_ZINC_'],
        'results_dict_greedy_bl_ZINC': results.get('results_dict_greedy_bl_ZINC_'),
        'failed_bl_ZINC': results.get('failed_bl_ZINC_'),
        'avg_tani_greedy_bl_ZINC': results['avg_tani_greedy_bl_ZINC_'],
        'results_dict_ZINC_greedy_bl': results.get('results_dict_ZINC_greedy_bl_'),
        'total_results_bl_ZINC': results['total_results_bl_ZINC_'],
        'corr_sampleing_prob_bl_ZINC': results['corr_sampleing_prob_bl_ZINC_'],
        'results_dict_bl_ZINC': results['results_dict_bl_ZINC_'],
    }
    save_data_with_datetime_index(variables_to_save, config.pkl_save_folder, "before_sim_data", idx)

def save_data_with_datetime_index(data, base_folder: str, name: str, idx) -> None:
    """Save data with datetime index"""
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_datetime}_{name}_{idx}.pkl"
    os.makedirs(base_folder, exist_ok=True)
    file_path = os.path.join(base_folder, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data saved to: {file_path}")

def test_pretrained_model_on_sim_data_before(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, idx):
    """Test pretrained model on simulated data"""
    MW_filter, greedy_full = True, False
    
    print("prepare_data")
    config = prepare_data(config, chunk)
    print("generate_simulated_data")
    config = generate_simulated_data(config, IR_config)

    print("load_model_and_data")
    model_MMT, val_dataloader, val_dataloader_multi = load_model_and_data(config, stoi, stoi_MF)

    print("run_model_analysis")
    try:
        prob_dict_results_1c_, results_dict_1c_ = mrtf.run_model_analysis(config, model_MMT, val_dataloader_multi, stoi, itos)
    except Exception as e:
        print(f"Error in model analysis: {e}")

    results = test_model_performance(config, model_MMT, val_dataloader, val_dataloader_multi, stoi, itos, stoi_MF, itos_MF)

    save_results_before(results, config, idx)

    return config

def run_base_model(chunk_size, config, IR_config, stoi, itos, stoi_MF, itos_MF):
    """Main function to run the base model on chunks"""
    chunks = split_dataset(config, chunk_size)
    config.model_save_dir = config.pkl_save_folder
    model_save_dir_backup = config.model_save_dir
    original_checkpoint_path = config.checkpoint_path

    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx+1} of {len(chunks)}")
        # if chunk_idx < 724:
        #     continue
        chunk_folder = create_chunk_folder(config, chunk_idx)
        config.current_chunk_folder = chunk_folder
            
        config.blank_percentage = 0
        config = test_pretrained_model_on_sim_data_before(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, f"{chunk_idx}_{0}")
        print(config.csv_1H_path_SGNN)

def main():
    """Main function to initialize and run the pipeline"""
    print("=" * 50)
    print("MultiModalSpectralTransformer Analysis Pipeline")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Script directory: {script_dir}")
    print(f"PyTorch CUDA devices: {torch.cuda.device_count()}")
    
    # Initialize configurations
    config, IR_config = initialize_configs()
    
    print("\nConfiguration initialized successfully!")
    print(f"Model save directory: {config.model_save_dir}")
    print(f"Checkpoint path: {config.checkpoint_path}")
    print(f"Data size: {config.data_size}")
    print(f"Execution type: {config.execution_type}")
    
    # Override specific settings for the experiment (matching your original script)
    config.SGNN_csv_gen_smi = os.path.join(project_root, "data/PubChem_dataset/val_data_0_250_x1000/ML_NMR_2M_XL_1H_V1_test_f_0_250_x1000.csv")
    config.pkl_save_folder = os.path.join(project_root, "experiments/baseline_PC_ZINC/PC_0_250")
    config.checkpoint_path = os.path.join(project_root, "models/mmst/V8i_MMT_Drop4/MultimodalTransformer_time_1710027004.1571195_Loss_0.112.ckpt")
    
    config.data_size = 1000
    config.execution_type = "test_performance"
    config.multinom_runs = 10
    config.temperature = 1
    
    # Create necessary directories
    os.makedirs(config.pkl_save_folder, exist_ok=True)
    
    # Parameters
    greedy_full = False
    MW_filter = True
    chunk_size = 1
    
    print("\nStarting model analysis...")
    print(f"Input CSV: {config.SGNN_csv_gen_smi}")
    print(f"Results folder: {config.pkl_save_folder}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Chunk size: {chunk_size}")
    
    # Run the analysis
    run_base_model(chunk_size, config, IR_config, stoi, itos, stoi_MF, itos_MF)
    
    return config, IR_config, stoi, itos, stoi_MF, itos_MF

if __name__ == "__main__":
    config, IR_config, stoi, itos, stoi_MF, itos_MF = main()