import matplotlib.pyplot as plt
# Core libraries
import json
import os
import random
import glob
import pickle
from datetime import datetime
import tempfile
import copy


# Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Machine learning and data visualization
import matplotlib.pyplot as plt
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

# tqdm for progress bars
from tqdm.autonotebook import tqdm

# Weights & Biases for experiment tracking
import wandb

# Miscellaneous
from argparse import Namespace
from IPython.display import HTML, SVG

# Setting up environment
torch.cuda.device_count()
#wandb.login()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Descriptors import MolWt
from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from tqdm import tqdm
#import seaborn as sns

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import os
import random

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

import utils_MMT.clip_functions_v15_4 as cf #
import utils_MMT.MT_functions_v15_4 as mtf # is different compared to V14_1
import utils_MMT.validate_generate_MMT_v15_4 as vgmmt #
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm #
import utils_MMT.clustering_visualization_v15_4 as cv #
import utils_MMT.plotting_v15_4 as pt #
import utils_MMT.execution_function_v15_4 as ex #
import utils_MMT.train_test_functions_pl_v15_4 as ttf
import utils_MMT.ir_simulation_v15_4 as irs
import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.mmt_result_test_functions_15_4 as mrtf


def load_json_dics():
    with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/itos.json', 'r') as f:
        itos = json.load(f)
    with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/stoi.json', 'r') as f:
        stoi = json.load(f)

    with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/stoi_MF.json', 'r') as f:
        stoi_MF = json.load(f)
    with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/itos_MF.json', 'r') as f:
        itos_MF = json.load(f)    
    return itos, stoi, stoi_MF, itos_MF
    
itos, stoi, stoi_MF, itos_MF = load_json_dics()
rand_num = str(random.randint(1, 10000000))

IR_config_dict = {
    "gpu": list(range(torch.cuda.device_count())),  # Default value is None, should be one of the available GPU indices
    "test_path": ["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/chemprop-IR/ir_models_data/solvation_example/solvation_spectra.csv"],  # Default value is None
    "use_compound_names": [False],  # Default is False
    "preds_path": ["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/chemprop-IR/ir_models_data/ir_preds_test_2.csv"],  # Default value is None
    #"checkpoint_dir": ["/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/chemprop-IR/ir_models_data/computed_model/model_files"],  # Default value is None
    "checkpoint_dir": ["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/chemprop-IR/ir_models_data/experiment_model/model_files"],  # Default value is None
    "spectra_type": ["experimental"],  # ["experimental", "simulated"] Default value is None
    "spectra_type_nr": [0],  # 0-4 Default value is None
    
    "checkpoint_path": [None],  # Default value is None
    "batch_size": [64],  # Default is 50
    "no_cuda": [False],  # Default is False
    "features_generator":[None],  # Default value is None, should be one of the available features generators
    "features_path": [None],  # Default value is None
    #"features_path": [["/projects/cc/knlr326/2_git_repos/chemprop-IR/ir_models_data/solvation_example/solvation_phases.csv" ]],  # Default value is None
    "max_data_size": [100],  # Default value is None
    "ensemble_variance": [False],  # Default is False
    "ensemble_variance_conv": [0.0],  # Default is 0.0
    #"dataset_type":["spectra"]
    }

hyperparameters = {
    # General project information
    "project": ["Improv_Cycle_v1"],  # Name of the project for wandb monitoring
    "ran_num":[rand_num],
    #"random_seed":[42], # random_seed
    "device": ["cuda"], # device on which training takes place
    "gpu_num":[1], # number of GPUs for training with pytorch lightning
    "num_workers":[4], # Needs to stay 1 otherwise code crashes - ToDO
    "data_type":["sgnn"], #["sgnn", "exp", "acd", "real", "inference"], Different data types to select
    "execution_type":["validate_MMT"], #[ "plot_similarities", "simulate_real", "test_performance", "SMI_generation_MMT", "SMI_generation_MF", "data_generation", "transformer_training","transformer_improvement", "clip_training", "clip_improvement", "validate_MMT"] # different networks to select for training
    "syn_data_simulated": [False],  # For the improvment cycle a ticker that shows whether data has been simulated or not.
    "training_type":["clip"], #["clip","transformer"] # different networks to select for training

    # Encoding dicts
    "itos_path":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/itos.json"],
    "stoi_path":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/stoi.json"],
    "itos_MF_path":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/itos_MF.json"],
    "stoi_MF_path":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/stoi_MF.json"],
    
    ### Data settings
    "input_dim_1H":[2], # Imput dimensions of the 1H data
    "input_dim_13C": [1], # Imput dimensions of the 13C data
    "input_dim_HSQC": [2], # Imput dimensions of the HSQC data
    "input_dim_COSY": [2],  # Imput dimensions of the COSY data
    "input_dim_IR": [1000],  # Imput dimensions of the IR data
    "MF_vocab_size": [len(stoi_MF)],  # New, size of the vocabulary for molecular formulas
    "MS_vocab_size": [len(stoi)],  # New, size of the vocabulary for molecular formulas
    "tr_te_split":[0.9], # Train-Test split
    "padding_points_number":[64], # Padding number for the embedding layer into the network
    "data_size": [1000], # number of datapoints for the training 3975764/1797828
    "test_size": [10], # number of datapoints for the training 3975764
    "model_save_dir": ["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/test"], # Folder where networks are saved
    "ML_dump_folder": ["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/dump"], # a folder where intermediate files for the SGNN network are generated
    "model_save_interval": [10000], # seconds passed until next model is saved
    
    # Option 1 SGNN
    "use_real_data":[False], #[True, False]
    "ref_data_type":["1H"], #["1H","13C","HSQC","COSY","IR"]
    "csv_train_path": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_comb_train_V8.csv'], # To keep a reference of the compounds that it was trained on
    "csv_1H_path_SGNN": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_comb_train_V8.csv'],
    "csv_13C_path_SGNN": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_13C.csv'],    
    "csv_HSQC_path_SGNN": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_HSQC.csv'],    
    "csv_COSY_path_SGNN": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_COSY.csv'],      
    "csv_IR_MF_path": [''],     #571124
    "csv_path_val": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_comb_test_V8.csv'], #63459   
    #"IR_data_folder": ["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/IR_spectra_NN"],
    "IR_data_folder": [""],

   # "pickle_file_path": ["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_comb_V8_938756.pkl"],
    "pickle_file_path": ["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_comb_test_V8_355655.pkl"],
    
    "dl_mode": ['val'], #["val","train"]   
    "isomericSmiles": [False], # whether stereochemistry is considered or not
    
    # Option 2 exp
    #"exp_path": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/9_ZINC_250k/missing_ZINC_files.csv'], #63459   

     # Option 2 ACD
    #"csv_path_1H_ACD": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/9_ZINC_250k/1H_ZINC_XL_v3.csv'],
    #"data_folder_HSQC_ACD": ["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/9_ZINC_250k/zinc250k"],
    # Option 3 real
    "comparision_number": [1000],  # With how many of the training examples should it be compared with in a t-SNE plot
    "vector_db": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/smiles_fingerprints_train_4M_v1.csv'],    
    "secret_csv_SMI_vectors": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/25_Test_Improvement_cycle/1_Test_ZINC_250/test_32_zinc250_vec_db.csv'],    
    "secret_csv_SMI_targets": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/25_Test_Improvement_cycle/1_Test_ZINC_250/test_32_zinc250.csv'],    
    "secret_csv_SMI_sim_searched": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/25_Test_Improvement_cycle/1_Test_ZINC_250/test_32_zinc250.csv'],    
    "csv_SMI_targets": ['/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/25_Test_Improvement_cycle/1_Test_ZINC_250/test_32_zinc250_single_target_919.csv'],
    "csv_1H_path_REAL": [''],
    "csv_13C_path_REAL": [''],    
    "csv_HSQC_path_REAL": [''],    
    "csv_COSY_path_REAL": [''],    
    #"pkl_path_HSQC_real": [""],
    # noising HSQC data
    #"noising_HSQC":[False],
    #"noising_peaks_file":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v1/noise_peaks_norm_4.pkl"],
    #"noising_dist_file":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v1/noise_num_list_norm_4.pkl"],

    #### Transformer Settings ####
    # Training and model settings
    "training_mode":["1H_13C_HSQC_COSY_IR_MF_MW"], #["edding_src_1H = torch.zeros((feature_dim, current_ba"], Modalities selected for training
    "blank_percentage":[0.0], # percentage of spectra that are blanked out during training for better generalizability of the network to various datatypes
    "batch_size":[64], # number needs to be the same as number of GPUs 
    "num_epochs": [10], # number of epochs for training
    "lr_pretraining": [1e-4], # Pretraining learning rate
    "lr_finetuning": [5e-5], # Finetuning learning rate
    "load_model": [True], # if model should be loaded from path
    "checkpoint_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/V8_MMT_MW_Drop/MultimodalTransformer_time_1704760608.6927748_Loss_0.137.ckpt"], #V8


    "save_model": [True], # if model should be saved
    # Model architecture
    "in_size": [len(stoi)],
    "hidden_size": [128],
    "out_size": [len(stoi)],
    "num_encoder_layers": [6], #8
    "num_decoder_layers": [6], #8
    "num_heads": [16], #8  ### number of attention heads
    "forward_expansion": [4], #4
    "max_len": [128], # maximum length of the generated sequence
    "drop_out": [0.1],
    "fingerprint_size": [512], # Dimensions of encoder output for CLIP contrastive training    
    #"track_metrics":[True],
    "gen_SMI_sequence":[True], # If the model generates a sequence with the SMILES current model for evaluation
    "sampling_method":["mix"], # weight_mol_weight ["multinomial", "greedy". "mix"]  
    "training_setup":["pretraining"], # ["pretraining","finetuning"]
    "smi_randomizer":[False], # if smiles are randomized or canonical during training
    ### SGNN Feedback
    "sgnn_feedback":[False], # if SGNN generates 1H and 13C spectrum on the fly on the generated smiles -> "gen_SMI_sequence":[True]
    "matching":["HungDist"], #["MinSum","EucDist","HungDist"], # HSQC point matching technique used
    "padding":["NN"], # ["Zero","Trunc","NN"], # HSQC padding technique used -> see publication: XXX
    # Weight feedback
    "train_weight_min":[None], # Calculate on the fly - Used for the weight loss calculation for scaling
    "train_weight_max":[None], # Calculate on the fly - Used for the weight loss calculation for scaling
    # Training Loss Weighting options
    #"symbol_reward_weight": [0.1], # loss weight if considered to contribute to loss function
    "weight_validity": [0.0], # up to 1
    "weight_SMI": [1.0], # up to 1
    #"weight_MF": [1.0], # up to 1
    "weight_FP": [0.0], # up to 1
    "weight_MW": [0], # up to 100
    "weight_sgnn": [0.0], # up to 10
    "weight_tanimoto": [0.0], # up to 1
    "change_loss_weights":[False], # if selected the weights get ajusted along the training
    "increment":[0.01], # increment on how much it gets ajusted during training -> TODO
    "batch_frequency":[10000], # Frequency how often it gets ajusted -> TODO
    
    ### For Validation
    "beam_size": [1],  
    "multinom_runs": [1], 
    "temperature":[1],
    "gen_len":[64],
    "pkl_save_folder":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/pkl_save_folder"],
    
    ### Molformer options 
    "MF_max_trails":[500],
    "MF_tanimoto_filter":[0.1],
    "MF_filter_higher":[1], # False = 0 True = 1
    "MF_delta_weight":[5],
    "MF_generations":[30],
    "MF_model_path":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/deep-molecular-optimization/experiments/trained/Alessandro_big/weights_pubchem_with_counts_and_rank_sanitized.ckpt"],
    "MF_vocab":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/deep-molecular-optimization/experiments/trained/Alessandro_big/vocab_new.pkl"],
    "MF_csv_source_folder_location":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/deep-molecular-optimization/data/MMP"],
    "MF_csv_source_file_name":["test_selection_2"],
    "MF_methods":["MMP"], #["MMP", "scaffold", "MMP_scaffold"],    
    "max_scaffold_generations":[10], #
    ### MMT batch generation
    "MMT_batch":[32], # how big is the batch of copies of the same inputs that is processed by MMT 
    "MMT_generations":[4], # need to be multiple of MMT_batch -> number of valid generated molecules
    #------------------------
    "n_samples":[10], # number of molecules that should be processed for data generation - needs to be smaller than dataloader size
    "gen_mol_csv_folder_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2"], # number of molecules that should be processed for data generation - needs to be smaller than dataloader size
    
    ### Fine-tuning improvement options
    "train_data_blend":[0], # how many additional molecules should be added to the new dataset from the original training dataset
    "train_data_blend_CLIP":[1000], # how many additional molecules should be added to the new dataset from the original training dataset
    
    ### Data generation SGNN -> 1H, 13C, HSQC, COSY
    "SGNN_gen_folder_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/dump_2"],
    "SGNN_csv_gen_smi":["/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/deep-molecular-optimization/data/MMP/test_selection_1.csv"],
    "SGNN_size_filter":[550],
    "SGNN_csv_save_folder":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2"],
    "IR_save_folder":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/IR_data"],
    
    #################################################
    #### LEGACY parameters for other expeirments ####
    #################################################
    #### CLIP Settings ####
    ### ChemBerta
    "model_version":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v1/Chemberta_source"],   # Source of pretrained chemberta from paper
    "CB_model_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v1/Large_300_15.pth"], # path to pretrained Chemberta model
    "num_class":[1024], #
    "num_linear_layers":[0], # number of linear layers in architecture before num_class output
    "use_dropout":[True],
    "use_relu":[False],
    "loss_fn":["BCEWithLogitsLoss"], #"MSELoss", "BCELoss", 
    "CB_embedding": [1024], #1024
    # PCA
    "fp_dim_reduction":[False], #True
    "pca_components":[300],  
    #"CB_model_name": ["Large_300_15"],

    ### Multimodal Transformer
    "MT_model_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/V8_MMT_MW2_Drop/MultimodalTransformer_time_1706856620.3718672_Loss_0.202.pth"],  # path to pretrained Multimodal Transformer model  
    #"MT_model_name": ["SpectrumBERT_PCA_large_3.6"],
    "MT_embedding": [512], #512
    ### Projection Head
    "projection_dim": [512],
    "dropout": [0.1],
    
    #CLIP
    # Dataloader settings
    "similarity_threshold":[0.6], # Filtere that selects just molecules with a tanimotosimilarity higher than that number
    "max_search_size":[10000], # Size of the data that will be searched to find the similar molecules  # 100000
    "weight_delta":[50], # Filter to molecules with a +/- delta weight of that numbeTraceback (most recent call last):
    "CLIP_batch_size":[128],  #,64,128,256 ### batch size for the CLIP training
    "CLIP_NUM_EPOCHS": [10],    # Number of training epochs
    
    ### Train parameters
    ### CLIP Model   
    "CLIP_temperature": [1],
    #"CB_projection_lr": [1e-3], # projection head learning rate for Chemberta
    "MT_projection_lr": [1e-3], # projection head learning rate for Multimodal Transfomer
    "CB_lr": [1e-4], # Chemberta Learning Rate
    "MT_lr": [1e-5], # Multimodal Transfomer Learning Rate
    "weight_decay": [1e-3], # Weight decay for projection heads -> TODO why just on those
    "patience": [1],   # not integrated yet
    "factor": [0.8],   # not integrated yet
    "CLIP_continue_training":[True],
    "CLIP_model_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/V8_modalities_CLIP_1_dot_product/MultimodalCLIP_Epoch_9_Loss0.096.ckpt"],   
    "CLIP_model_save_dir":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/test_CLIP"],
    
    ### BLIP Model
    "BLIP_temperature": [1],
    "Qformer_lr":[1e-4],
    "Qformer_CB_lr":[1e-4],
    "Qformer_MT_lr":[1e-4],
    "BLIP_continue_training":[True],
    # "BLIP_model_path":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/test_BLIP_1M/model_BLIP-epoch=03-loss=2.54_v0.ckpt"],   
    # "BLIP_model_save_dir":["/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/test_BLIP_1M"],
    
    }





def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f)

def load_config(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None    

def parse_arguments(hyperparameters):
    # Using dictionary comprehension to simplify your code
    parsed_args = {key: val[0] for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)


config = parse_arguments(hyperparameters)
ir_config_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/utils_MMT/ir_config_V8.json'
save_config(IR_config_dict, ir_config_path)
IR_config_dict = load_config(ir_config_path)
IR_config = parse_arguments(IR_config_dict)
irs.modify_predict_args(IR_config)


config_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/utils_MMT/config_V8.json'
save_config(hyperparameters, config_path)
config_dict = load_config(config_path)
config = parse_arguments(config_dict)

import os
from typing import List, Dict, Any, Union, Tuple
import pandas as pd
from datetime import datetime
import pickle
import re
import tempfile



def process_pkl_files(folder_path, file_type, ranking_method):
    pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.endswith('.pkl') and file_type in f]
    
    all_rankings = defaultdict(list)
    
    for file_path in pkl_files:
        file_data = load_data(file_path)
        ranked_molecules = rank_molecules_in_file(file_data, ranking_method)
        
        for molecule in ranked_molecules:
            trg_smi = molecule[0]
            all_rankings[trg_smi].append(molecule)
    
    return all_rankings


def split_dataset(config, chunk_size: int) -> List[pd.DataFrame]:
    df = pd.read_csv(config.SGNN_csv_gen_smi)
    return [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def create_chunk_folder(config, idx: int) -> str:
    base_dir = config.model_save_dir
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_folder_name = f"chunk_{idx:03d}_{current_datetime}"
    chunk_folder_path = os.path.join(base_dir, chunk_folder_name)
    
    os.makedirs(chunk_folder_path, exist_ok=True)
    print(f"Created folder for chunk {idx}: {chunk_folder_path}")
    
    return chunk_folder_path

def test_pretrained_model_on_sim_data_before(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, idx):
    MW_filter, greedy_full = True, False
    
    print("prepare_data")
    config = prepare_data(config, chunk)
    print("generate_simulated_data")
    config = generate_simulated_data(config, IR_config)

    print("load_model_and_data")
    model_MMT, val_dataloader, val_dataloader_multi = load_model_and_data(config, stoi, stoi_MF)

    print("run_model_analysis")
    prob_dict_results_1c_, results_dict_1c_ = mrtf.run_model_analysis(config, model_MMT, val_dataloader_multi, stoi, itos)

    results = test_model_performance(config, model_MMT, val_dataloader, val_dataloader_multi, stoi, itos, stoi_MF, itos_MF)

    save_results_before(results, config, idx)

    return config

def prepare_data(config: Any, chunk: pd.DataFrame) -> Any:
    chunk_csv_path = os.path.join(config.pkl_save_folder, "SGNN_csv_gen_smi.csv")
    chunk.to_csv(chunk_csv_path)
    config.SGNN_csv_gen_smi = chunk_csv_path 
    config.data_size = len(chunk)
    return config

def generate_simulated_data(config: Any, IR_config: Any) -> Any:
    config.execution_type = "data_generation"
    if config.execution_type == "data_generation":
        print("\033[1m\033[31mThis is: data_generation\033[0m")
        #import IPython; IPython.embed();

        config = ex.gen_sim_aug_data(config, IR_config)
        backup_config_paths(config)
    return config

def backup_config_paths(config: Any) -> None:
    config.csv_1H_path_SGNN_backup = copy.deepcopy(config.csv_1H_path_SGNN)
    config.csv_13C_path_SGNN_backup = copy.deepcopy(config.csv_13C_path_SGNN)
    config.csv_HSQC_path_SGNN_backup = copy.deepcopy(config.csv_HSQC_path_SGNN)
    config.csv_COSY_path_SGNN_backup = copy.deepcopy(config.csv_COSY_path_SGNN)
    config.IR_data_folder_backup = copy.deepcopy(config.IR_data_folder)

def save_results_before(results: Dict[str, Any], config: Any, idx: int) -> None:
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

def create_run_folder(chunk_folder, idx):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{idx}_{current_datetime}"
    run_folder_path = os.path.join(chunk_folder, run_folder_name)
    
    os.makedirs(run_folder_path, exist_ok=True)
    print(f"Created folder for run {idx}: {run_folder_path}")
    
    return run_folder_path

def fine_tune_model_aug_mol(config, stoi, stoi_MF, chunk, idx):
    #import IPython; IPython.embed();
    config, all_gen_smis, aug_mol_df = generate_augmented_molecules_from_aug_mol(config, chunk, idx)
    
    config.parent_model_save_dir = config.model_save_dir
    config.model_save_dir = config.current_run_folder 
    
    if config.execution_type == "transformer_improvement":
        print("\033[1m\033[31mThis is: transformer_improvement, sim_data_gen == TRUE\033[0m")
        config.training_setup = "pretraining"
        mtf.run_MMT(config, stoi, stoi_MF)
    
    config.model_save_dir = config.parent_model_save_dir
    #config = ex.update_model_path(config)

    return config, aug_mol_df, all_gen_smis


def generate_augmented_molecules_from_aug_mol(config, chunk, idx):
    #import IPython; IPython.embed();

    ############# THis is just relevant for the augmented molecules #############
    chunk.rename(columns={'SMILES': 'SMILES_orig', 'SMILES_regio_isomers': 'SMILES'}, inplace=True)
    #############################################################################
    
    script_dir = os.getcwd()
    
    base_path = os.path.abspath(os.path.join(script_dir, 'deep-molecular-optimization'))

    csv_file_path = f'{base_path}/data/MMP/test_selection_2.csv'
    chunk.to_csv(csv_file_path, index=False)
    print(f"CSV file '{csv_file_path}' created successfully.")

    config.data_size = len(chunk)
    config.n_samples = config.data_size

    config, results_dict_MF = generate_smiles_mf(config)

    combined_list_MF = process_generated_smiles(results_dict_MF, config)

    all_gen_smis = filter_and_combine_smiles(combined_list_MF)

    aug_mol_df = create_augmented_dataframe(all_gen_smis)

    config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)

    config = ex.gen_sim_aug_data(config, IR_config)
    config.execution_type = "transformer_improvement"

    return config, all_gen_smis, aug_mol_df


def fine_tune_model(config, stoi, stoi_MF, chunk, idx):
    """
    Fine-tune the model on a chunk of data.
    """
    config, aug_mol_df, all_gen_smis = generate_augmented_molecules(config, chunk, idx)
    
    config.parent_model_save_dir = config.model_save_dir
    new_model_save_dir = create_model_save_dir(config.parent_model_save_dir, idx)
    config.model_save_dir = new_model_save_dir
    
    # Fine-tune the model
    if config.execution_type == "transformer_improvement":
        print("\033[1m\033[31mThis is: transformer_improvement, sim_data_gen == TRUE\033[0m")
        config.training_setup = "pretraining"
        mtf.run_MMT(config, stoi, stoi_MF)
        
    #config = ex.update_model_path(config)
    config.model_save_dir = config.parent_model_save_dir
    
    return config, aug_mol_df, all_gen_smis

def generate_augmented_molecules(config, chunk, idx):
    #import IPython; IPython.embed();
    script_dir = os.getcwd()
    
    base_path = os.path.abspath(os.path.join(script_dir, 'deep-molecular-optimization'))

    csv_file_path = f'{base_path}/data/MMP/test_selection_2.csv'
    chunk.to_csv(csv_file_path, index=False)
    print(f"CSV file '{csv_file_path}' created successfully.")

    config.data_size = len(chunk)
    config.n_samples = config.data_size

    config, results_dict_MF = generate_smiles_mf(config)

    combined_list_MF = process_generated_smiles(results_dict_MF, config)

    all_gen_smis = filter_and_combine_smiles(combined_list_MF)

    aug_mol_df = create_augmented_dataframe(all_gen_smis)

    config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)

    config = ex.gen_sim_aug_data(config, IR_config)
    config.execution_type = "transformer_improvement"

    return config, all_gen_smis, aug_mol_df


def generate_smiles_mf(config):
    print("\033[1m\033[31mThis is: SMI_generation_MF\033[0m")
    return ex.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)

def process_generated_smiles(results_dict_MF, config):
    results_dict_MF = {key: value for key, value in results_dict_MF.items() if not hf.contains_only_nan(value)}
    for key, value in results_dict_MF.items():
        results_dict_MF[key] = hf.remove_nan_from_list(value)

    combined_list_MF, _, _, _ = cv.plot_cluster_MF(results_dict_MF, config)
    return combined_list_MF

def filter_and_combine_smiles(combined_list_MF):
    print("\033[1m\033[31mThis is: combine_MMT_MF\033[0m")
    all_gen_smis = combined_list_MF
    all_gen_smis = [smiles for smiles in all_gen_smis if smiles != 'NAN']

    val_data = pd.read_csv(config.csv_path_val)
    all_gen_smis = mrtf.filter_smiles(val_data, all_gen_smis)
    return all_gen_smis

def create_augmented_dataframe(all_gen_smis):
    length_of_list = len(all_gen_smis)
    random_number_strings = [f"GT_{str(i).zfill(7)}" for i in range(1, length_of_list + 1)]
    return pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})

def setup_data_paths(config):
    base_path_acd = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/37_Richard_ACD_sim_data/"
    config.csv_1H_path_ACD = f"{base_path_acd}ACD_1H_with_SN_filtered_v3.csv"
    config.csv_13C_path_ACD = f"{base_path_acd}ACD_13C_with_SN_filtered_v3.csv"
    config.csv_HSQC_path_ACD = f"{base_path_acd}ACD_HSQC_with_SN_filtered_v3.csv"
    config.csv_COSY_path_ACD = f"{base_path_acd}ACD_COSY_with_SN_filtered_v3.csv"
    config.IR_data_folder_ACD = f"{base_path_acd}IR_spectra"
    
    base_path_exp = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/36_Richard_43_dataset/experimenal_data/"
    config.csv_1H_path_exp = f"{base_path_exp}real_1H_with_AZ_SMILES_v3.csv"
    config.csv_13C_path_exp = f"{base_path_exp}real_13C_with_AZ_SMILES_v3.csv"
    config.csv_HSQC_path_exp = f"{base_path_exp}real_HSQC_with_AZ_SMILES_v3.csv"
    config.csv_COSY_path_exp = f"{base_path_exp}real_COSY_with_AZ_SMILES_v3.csv"
    config.IR_data_folder_exp = f"{base_path_exp}IR_data"
    return config

def test_model_on_datasets(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, composite_idx, aug_mol_df, all_gen_smis):
    checkpoint_path_backup = config.checkpoint_path    
    for data_type in ['exp', 'sim', 'ACD', ]:
        print(f"Testing on {data_type} data")
        config.pickle_file_path = ""
        config.training_mode = "1H_13C_HSQC_COSY_IR_MF_MW"
        config = test_on_data(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, composite_idx, data_type, aug_mol_df, all_gen_smis)
    config.checkpoint_path = checkpoint_path_backup
    return config

def test_on_data(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, composite_idx, data_type, aug_mol_df, all_gen_smis):
    if data_type == 'sim':
        restore_backup_configs(config)
    else:
        sample_ids = chunk['sample-id'].tolist()
        process_spectrum_data(config, sample_ids, data_type)
    #import IPython; IPython.embed();

    update_config_settings(config)
    last_checkpoint = get_last_checkpoint(config.current_run_folder)
    config.checkpoint_path = last_checkpoint
    
    model_MMT, val_dataloader, val_dataloader_multi = load_model_and_data(config, stoi, stoi_MF)
    
    prob_dict_results_1c_, results_dict_1c_ = mrtf.run_model_analysis(config, model_MMT, val_dataloader_multi, stoi, itos)

    results = test_model_performance(config, model_MMT, val_dataloader, val_dataloader_multi,
                                     stoi, itos, stoi_MF, itos_MF)
    
    if data_type == 'sim':
        results['aug_mol_df'] = aug_mol_df
        results['all_gen_smis'] = all_gen_smis
    
    save_results_acd_exp(results, config, data_type, composite_idx)
    return config

def restore_backup_configs(config):
    config.csv_1H_path_SGNN = config.csv_1H_path_SGNN_backup
    config.csv_13C_path_SGNN = config.csv_13C_path_SGNN_backup
    config.csv_HSQC_path_SGNN = config.csv_HSQC_path_SGNN_backup
    config.csv_COSY_path_SGNN = config.csv_COSY_path_SGNN_backup
    config.IR_data_folder = config.IR_data_folder_backup 
    config.csv_path_val = config.csv_1H_path_SGNN_backup
    config.pickle_file_path = ""

def process_spectrum_data(config: Any, sample_ids: List[str], data_type: str) -> None:
    spectrum_types = ['1H', '13C', 'HSQC', 'COSY']
    for spectrum in spectrum_types:
        csv_path = getattr(config, f'csv_{spectrum}_path_{data_type}')
        df_data = pd.read_csv(csv_path)
        df_data['sample-id'] = df_data['AZ_Number']
        data = select_relevant_samples(df_data, sample_ids)
        dummy_path, config = save_and_update_config(config, data_type, spectrum, data)
        print(f"Saved {spectrum} data to: {dummy_path}")
    if data_type == "ACD" or data_type == "sim":
        config.IR_data_folder = config.IR_data_folder_backup 
    elif  data_type == "exp":
        config.IR_data_folder = config.IR_data_folder_exp 

    
    
def select_relevant_samples(df: pd.DataFrame, sample_ids: List[str]) -> pd.DataFrame:
    return df[df['sample-id'].isin(sample_ids)]

def save_and_update_config(config, data_type: str, spectrum_type: str, data: pd.DataFrame) -> Tuple[str, Any]:
    temp_dir = tempfile.mkdtemp()
    dummy_path = os.path.join(temp_dir, f"{data_type}_{spectrum_type}_selected_samples.csv")
    
    data.to_csv(dummy_path, index=False)
    
    config_key = f'csv_{spectrum_type}_path_SGNN'
    setattr(config, config_key, dummy_path)
    
    return dummy_path, config

def update_config_settings(config: Any) -> None:
    config.csv_path_val = config.csv_1H_path_SGNN
    config.pickle_file_path = ""

def get_last_checkpoint(model_folder: str) -> str:
    checkpoints = [f for f in os.listdir(model_folder) if f.endswith('.ckpt')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_folder}")
    
    last_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(model_folder, x)))
    return os.path.join(model_folder, last_checkpoint)

def load_model_and_data(config: Any, stoi: Dict, stoi_MF: Dict) -> Tuple[Any, Any, Any]:
    #import IPython; IPython.embed();

    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    model_MMT = mrtf.load_MMT_model(config)
    return model_MMT, val_dataloader, val_dataloader_multi

def test_model_performance(config: Any, model_MMT: Any, val_dataloader: Any, val_dataloader_multi: Any, 
                           stoi: Dict, itos: Dict, stoi_MF: Dict, itos_MF: Dict) -> Dict[str, Any]:
    print("\033[1m\033[31mThis is: test_performance\033[0m")
    
    MW_filter = True
    greedy_full = False
    
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

    print("avg_tani, avg_tani_greedy, corr_sampleing_prob'")
    print(results['avg_tani_bl_ZINC_'], results['avg_tani_greedy_bl_ZINC_'], results['corr_sampleing_prob_bl_ZINC_'])
    print("Greedy tanimoto results")
    rbgvm.plot_hist_of_results_greedy_new(results['results_dict_ZINC_greedy_bl_'])

    return results

def save_results_acd_exp(results: Dict[str, Any], config: Any, data_type: str, composite_idx: str) -> None:
    variables_to_save = {
        'avg_tani_bl_ZINC': results['avg_tani_bl_ZINC_'],
        'results_dict_greedy_bl_ZINC': results.get('results_dict_greedy_bl_ZINC_'),
        'failed_bl_ZINC': results.get('failed_bl_ZINC_'),
        'avg_tani_greedy_bl_ZINC': results['avg_tani_greedy_bl_ZINC_'],
        'results_dict_ZINC_greedy_bl': results.get('results_dict_ZINC_greedy_bl_'),
        'total_results_bl_ZINC': results['total_results_bl_ZINC_'],
        'corr_sampleing_prob_bl_ZINC': results['corr_sampleing_prob_bl_ZINC_'],
        'results_dict_bl_ZINC': results['results_dict_bl_ZINC_'],
        'checkpoint_path': config.checkpoint_path,
    }
    
    if data_type == 'sim':
        variables_to_save['aug_mol_df'] = results.get('aug_mol_df')
        variables_to_save['all_gen_smis'] = results.get('all_gen_smis')
    
    save_data_with_datetime_index(
        variables_to_save, 
        config.pkl_save_folder, 
        f"{data_type}_sim_data", 
        composite_idx
    )

def save_data_with_datetime_index(data: Any, base_folder: str, name: str, idx: Union[int, str]) -> None:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_datetime}_{name}_{idx}.pkl"
    os.makedirs(base_folder, exist_ok=True)
    file_path = os.path.join(base_folder, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data saved to: {file_path}")


import os
from typing import List, Dict, Any, Union, Tuple
import pandas as pd
from datetime import datetime
import pickle
import re
import tempfile

def load_model_and_data(config: Any, stoi: Dict, stoi_MF: Dict) -> Tuple[Any, Any, Any]:
    #import IPython; IPython.embed();

    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    model_MMT = mrtf.load_MMT_model(config)
    return model_MMT, val_dataloader, val_dataloader_multi

def run_base_model(chunk_size, config, IR_config, stoi, itos, stoi_MF, itos_MF):
    chunks = split_dataset(config, chunk_size)
    config.model_save_dir = config.pkl_save_folder
    model_save_dir_backup = config.model_save_dir
    original_checkpoint_path = config.checkpoint_path  # Store the original checkpoint path

    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx+1} of {len(chunks)}")
        
        chunk_folder = create_chunk_folder(config, chunk_idx)
        config.current_chunk_folder = chunk_folder
            
        config.blank_percentage = 0
        config = test_pretrained_model_on_sim_data_before(config, IR_config, stoi, itos, stoi_MF, itos_MF, chunk, f"{chunk_idx}_{0}")
        print(config.csv_1H_path_SGNN)




if __name__ == "__main__":
        
    config.SGNN_csv_gen_smi = '/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_comb_test_V8_1000.csv'
    config.pkl_save_folder = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/47_Anna_paper_data/Experiment_baseline_PC_ZINC/ZINC_250_350"
    config.checkpoint_path = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/1_old_models/models_v2/V8i_MMT_Drop4/MultimodalTransformer_time_1710027004.1571195_Loss_0.112.ckpt"

    config.data_size = 1000 # config.test_size # why would I do that? 
    #config.data_size = 4 # config.test_size # why would I do that? 
    config.execution_type = "test_performance"
    config.multinom_runs = 10
    #config.multinom_runs = 3
    config.temperature = 1
    greedy_full = False
    MW_filter = True
    #config.MF_generations = 10
    chunk_size = 1
    run_base_model(chunk_size, config, IR_config, stoi, itos, stoi_MF, itos_MF)
