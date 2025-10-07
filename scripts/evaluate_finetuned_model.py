#!/usr/bin/env python3
"""
Evaluation script for fine-tuned MultiModalSpectralTransformer models.
This script loads a fine-tuned model and evaluates its performance on the
corresponding full 1000-molecule PubChem test set.
Accepts the dataset name as a command-line argument.
"""

# Base imports
import json
import os
import pickle
import sys
from argparse import Namespace, ArgumentParser
from collections import defaultdict

# Scientific computing imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch and RDKit
import torch
from rdkit import Chem

# --- FIX: Correctly determine the project root directory ---
# The project root is now the directory where this script is located.
project_root = os.path.dirname(os.path.abspath(__file__))
# Add the utils directory to the Python path to ensure modules are found
sys.path.append(os.path.join(project_root, 'utils_MMT'))

# --- Import custom modules ---
try:
    import utils_MMT.MT_functions_v15_4 as mtf
    import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
    import utils_MMT.mmt_result_test_functions_15_4 as mrtf
except ImportError as e:
    print(f"FATAL: Could not import required custom modules: {e}")
    sys.exit(1)

def get_last_checkpoint(model_folder: str) -> str:
    """Finds the most recently saved .ckpt file in a directory."""
    checkpoints = [f for f in os.listdir(model_folder) if f.endswith('.ckpt')]
    if not checkpoints:
        raise ValueError(f"CRITICAL: No checkpoints found in the specified model folder: {model_folder}")
    
    # Sort by modification time to get the most recent file
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(model_folder, x)))
    last_checkpoint = checkpoints[-1]
    return os.path.join(model_folder, last_checkpoint)


def setup_paths_for_evaluation(set_name: str, config: Namespace) -> Namespace:
    """Configures all necessary paths based on the selected dataset."""
    print(f"\nConfiguring paths for dataset: {set_name}...")
    
    reviewer_exp_base_dir = os.path.abspath(os.path.join(project_root, "past_experiments/ChemXriv/Reviewer_Experiment_Global_Impact"))
    
    finetuned_model_dir = os.path.join(reviewer_exp_base_dir, set_name, "finetuned_model")
    try:
        config.checkpoint_path = get_last_checkpoint(finetuned_model_dir)
        print(f"  - Using fine-tuned model: {os.path.basename(config.checkpoint_path)}")
    except ValueError as e:
        print(e)
        sys.exit(1)

    if set_name == "PC_0_250_Da":
        data_folder = os.path.join(project_root, "data/PubChem_dataset/val_data_0_250_x1000")
        config.csv_1H_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_1H_V1_test_f_0_250_x1000.csv")
        config.csv_13C_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_13C_V1_test_0_250_x1000.csv")
        config.csv_HSQC_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_HSQC_V1_test_0_250_x1000.csv")
        config.csv_COSY_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_COSY_V1_test_0_250_x1000.csv")
    elif set_name == "PC_250_350_Da":
        data_folder = os.path.join(project_root, "data/PubChem_dataset/val_data_250_350_x1000")
        config.csv_1H_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_1H_V1_test_f_250_350_x1000.csv")
        config.csv_13C_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_13C_V1_test_250_350_x1000.csv")
        config.csv_HSQC_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_HSQC_V1_test_250_350_x1000.csv")
        config.csv_COSY_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_COSY_V1_test_250_350_x1000.csv")
    elif set_name == "PC_350_500_Da":
        data_folder = os.path.join(project_root, "data/PubChem_dataset/val_data_350_500_x1000")
        config.csv_1H_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_1H_V1_test_f_350_500_x1000.csv")
        config.csv_13C_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_13C_V1_test_350_500_x1000.csv")
        config.csv_HSQC_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_HSQC_V1_test_350_500_x1000.csv")
        config.csv_COSY_path_SGNN = os.path.join(data_folder, "ML_NMR_2M_XL_COSY_V1_test_350_500_x1000.csv")
    else:
        raise ValueError(f"Invalid dataset name provided: {set_name}")

    config.IR_data_folder = os.path.join(project_root, "data/PubChem_dataset/IR_data")
    config.csv_path_val = config.csv_1H_path_SGNN
    config.data_size = 1000
    config.pickle_file_path = ""
    
    print(f"  - Using test data from: {os.path.basename(data_folder)}")
    return config
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

def main():
    """Main function to run the evaluation pipeline."""
    parser = ArgumentParser(description="Evaluate a fine-tuned MMST model.")
    parser.add_argument("set_name", type=str, choices=["PC_0_250_Da", "PC_250_350_Da", "PC_350_500_Da"],
                        help="The name of the dataset to evaluate.")
    args = parser.parse_args()
    SET_TO_EVALUATE = args.set_name

    print("=" * 50)
    print("MMST Fine-Tuned Model Evaluation Pipeline")
    print(f"Executing for: {SET_TO_EVALUATE}")
    print("=" * 50)
    
    def load_json_dics(root_path):
        """Loads vocabulary JSON files using absolute paths."""
        print("\nLoading vocabulary files...")
        with open(os.path.join(root_path, 'itos.json'), 'r') as f:
            itos = json.load(f)
        with open(os.path.join(root_path, 'stoi.json'), 'r') as f:
            stoi = json.load(f)
        with open(os.path.join(root_path, 'stoi_MF.json'), 'r') as f:
            stoi_MF = json.load(f)
        with open(os.path.join(root_path, 'itos_MF.json'), 'r') as f:
            itos_MF = json.load(f)
        print("  - Vocabularies loaded successfully.")
        return itos, stoi, stoi_MF, itos_MF

    itos, stoi, stoi_MF, itos_MF = load_json_dics(project_root)
    
    config_path = os.path.join(project_root, 'utils_MMT', 'config_V8.json')
    print(f"Loading base configuration from: {config_path}")
    with open(config_path, 'r') as f:
        hyperparameters = json.load(f)
    config = Namespace(**{key: val[0] for key, val in hyperparameters.items()})
    
    config = setup_paths_for_evaluation(SET_TO_EVALUATE, config)
    
    config.multinom_runs = 10
    config.temperature = 1.0
    MW_filter = True
    
    print("\nLoading model and data...")
    model_MMT = mrtf.load_MMT_model(config)
    model_CLIP = mrtf.load_CLIP_model(config)
    val_dataloader_single = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    
    print(f"\nStarting evaluation on {config.data_size} molecules. This will take some time...")
    results = test_model_performance(config, model_MMT, val_dataloader_single, val_dataloader_multi,
                                          stoi, itos, stoi_MF, itos_MF)
                                     
    output_dir = os.path.join(project_root, "past_experiments/ChemXriv/Reviewer_Experiment_Global_Impact", SET_TO_EVALUATE)
    output_file_path = os.path.join(output_dir, f"evaluation_results_{SET_TO_EVALUATE}.pkl")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)
        
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print(f"All results for '{SET_TO_EVALUATE}' have been saved to:")
    print(output_file_path)
    print("="*50)


if __name__ == "__main__":
    main()