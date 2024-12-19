# Standard library imports
import ast
import base64
import csv
import datetime
import io
import json
import logging
import operator
import os
import pickle
import random
import shutil
import urllib.parse
from functools import reduce
from pathlib import Path
import io
from tqdm import tqdm

# Third-party imports
## Flask and related
from flask import Flask, render_template, send_file, request, redirect, flash, jsonify, session
from flask_socketio import SocketIO, emit

## Data processing and scientific computing
import numpy as np
import pandas as pd

## Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go

## Machine learning
from umap import UMAP

## Chemistry-related
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

## Image processing
from PIL import Image, ImageDraw, ImageFont

# Local imports
import functions as f
from MMT_import import *


app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necessary for session management
socketio = SocketIO(app)

script_dir = os.path.dirname(__file__)

# Define paths relative to the script's location
UPLOAD_FOLDER = os.path.abspath(os.path.join(script_dir, 'Upload_Folder'))
config_PATH = os.path.abspath(os.path.join(script_dir, 'config_V8.json'))

data_dict = {}
current_index = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
@app.route('/')
def index():
    state = session.get('state', {})
    return render_template('index.html', state=state)


@app.route('/next_molecule')
def next_molecule():
    global current_index
    if current_index < len(data_dict) - 1:
        current_index += 1
    return jsonify({"success": True})

@app.route('/prev_molecule')
def prev_molecule():
    global current_index
    if current_index > 0:
        current_index -= 1
    return jsonify({"success": True})


@app.route('/upload', methods=['GET', 'POST'])
def upload_files_individually():
    global data_dict
    if request.method == 'POST':
        uploaded_files = {
            '1H_Real': request.files.get('file_1H'),
            '13C_Real': request.files.get('file_13C'),
            'HSQC_Real': request.files.get('file_HSQC'),
            'COSY_Real': request.files.get('file_COSY'),
            'IR_Real': request.files.get('file_IR')
        }
        
        for nmr_type, file in uploaded_files.items():
            if file and file.filename:
                if not file.filename.endswith('.csv'):
                    print(f'Invalid file type for {nmr_type}. Please upload a CSV file.', 'error')
                    continue

                filename = file.filename
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)  # Save the uploaded file to the upload folder
                data_dict = f.parse_NMR_csv(file_path, nmr_type)

                config_file_path = config_PATH
                data = {}

                data[nmr_type] = file_path


                # Read the existing config file if it exists
                if os.path.exists(config_file_path):
                    with open(config_file_path, 'r') as json_file:
                        config_data = json.load(json_file)
                else:
                    config_data = {}

                # Update the config data with the new SMILES data
                config_data.update(data)

                # Write the updated config data back to the JSON file
                with open(config_file_path, 'w') as json_file:
                    json.dump(config_data, json_file, indent=4)

            else:
                print(f'No file selected for {nmr_type}', 'warning')

        return redirect('/')
    
    return render_template('upload.html')
    
 
@app.route('/molecule_image/', defaults={'index': None})
@app.route('/molecule_image/<int:index>')
def molecule_image(index):
    global data_dict, current_index

    sample_ids = data_dict['sample-id']
    
    size = (300, 400)

    if index is None:
        index = current_index
        size = (300, 300)

    if index < 0 or index >= len(sample_ids):
        return "Invalid molecule index", 404
 
    sample_id = sample_ids[index]
    molecule_smiles = data_dict['SMILES'][index]
    molecule = Chem.MolFromSmiles(molecule_smiles)

    if molecule is None:
        return "Invalid molecule structure", 500

    # Calculate molecular weight
    mol_weight = round(Descriptors.ExactMolWt(molecule), 2)

    # Generate molecule image
    img = Draw.MolToImage(molecule, size)

    # No need to convert to PIL Image, as img is already a PIL Image
    draw = ImageDraw.Draw(img)

    # Use a default font
    font = ImageFont.load_default()

    # Add sample ID and molecular weight to the image
    draw.text((10, 10), f"Sample ID: {sample_id}", fill=(0, 0, 0), font=font)
    draw.text((10, 30), f"MW: {mol_weight}", fill=(0, 0, 0), font=font)

    # Save the image to a byte stream
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

def plot_hist_of_results_combined(results_dict_greedy, results_dict_mns):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Greedy sampling plot
    tani_list_greedy = results_dict_greedy.get('tanimoto_sim', [])
    tani_list_greedy = [t for t in tani_list_greedy if t != 0]
    successful_greedy = len(tani_list_greedy)
    failed_greedy = len(results_dict_greedy.get('tanimoto_sim', [])) - successful_greedy

    ax1.hist(tani_list_greedy, bins=20, edgecolor='black')
    ax1.set_title(f'Greedy Sampled Tanimoto Similarity\n{successful_greedy} Successful, {failed_greedy} Failed Molecules')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')

    avg_tani_greedy = sum(tani_list_greedy) / len(tani_list_greedy) if tani_list_greedy else 0
    ax1.text(0.95, 0.95, f'Avg. Tanimoto: {avg_tani_greedy:.3f}', 
             verticalalignment='top', horizontalalignment='right',
             transform=ax1.transAxes, fontsize=10)

    # MNS sampling plot
    tani_list_mns = []
    for key in tqdm(results_dict_mns.keys()):
        sorted_list = results_dict_mns[key]
        third_element = sorted_list[0][0][4]
        tani_list_mns.append(third_element)

    ax2.hist(tani_list_mns, bins=20, edgecolor='black')
    ax2.set_title(f'MNS Tanimoto Similarity\nSample size {len(results_dict_mns)}')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')

    avg_tani_mns = sum(tani_list_mns) / len(tani_list_mns) if tani_list_mns else 0
    ax2.text(0.95, 0.95, f'Avg. Tanimoto: {avg_tani_mns:.3f}', 
             verticalalignment='top', horizontalalignment='right',
             transform=ax2.transAxes, fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close(fig)  # Close the figure to free up memory
    print_to_console("Plotted Molecule")

    return img_io



def get_path(config, spectrum_type):
    key_map = {
        "1H": "csv_1H_path_display",
        "13C": "csv_13C_path_display",
        "HSQC": "csv_HSQC_path_display",
        "COSY": "csv_COSY_path_display",
        "IR": "IR_data_folder_display"        
    }
    
    attr_name = key_map.get(spectrum_type)
    if attr_name is None:
        raise ValueError(f"Invalid spectrum type: {spectrum_type}")

    if hasattr(config, attr_name):
        return getattr(config, attr_name)
    else:
        raise AttributeError(f"Config does not have attribute: {attr_name}")
 

@app.route('/simulate/<path:SMILES_Path>', methods=['GET'])
def simulate(SMILES_Path):
    try:
        print_to_console("Function simulate: Start of Simulation")
        IR_config, config = load_configs()
        config.simulated = True
        config.SGNN_csv_gen_smi = "/"+ SMILES_Path
        save_updated_config(config, config.config_path)
        config = sim_and_display() ### UNCOMMENT FOR ACTUALLY TESTING THE CODE
        ### just for testing the code
        # config.csv_1H_path_display = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/data_1H_265916.csv"
        # config.csv_13C_path_display = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/data_13C_265916.csv"
        # config.csv_HSQC_path_display = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/data_COSY_265916.csv"
        # config.csv_COSY_path_display = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/data_HSQC_265916.csv"
        # config.IR_data_folder_display = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/24_SGNN_gen_folder_2/dump_2_265916_265916_265916_265916_265916"
        ##########################################################
        ### this is where you can get the spectra for plotting ###
        ##########################################################
        #plot_first_smiles(config.csv_1H_path_SGNN)
        print(config.csv_1H_path_display)
        print(config.csv_13C_path_display)
        print(config.csv_HSQC_path_display)
        print(config.csv_COSY_path_display)
        print(config.IR_data_folder_display)
        config.SGNN_csv_gen_smi = config.csv_1H_path_display
        save_updated_config(config, config.config_path)
        print_to_console("Function simulate: Simulation Succeeded")
        print("Succeeded")
        #import IPython; IPython.embed();

        return '', 204  # No content to return, but the request was successful
    except Exception as e:
        logger.error("Error test: %s", e)
        return str(e), 500  # Return the error message with a 500 status code



@app.route('/plot_nmr')
def plot_nmr():
    try:
        global current_index, data_dict

        nmr_type = request.args.get('nmr_type', session.get('nmr_type', '1H'))
        index_str = int(request.args.get('index', 0))
        try:
            index = int(index_str)
        except ValueError:
            index = 0  # Default to 0 if index is not a valid integer

        IR_data = {}
        nmr_type = request.args.get('nmr_type', session.get('nmr_type', '1H'))
        session['nmr_type'] = nmr_type

        IR_config, config_data = load_configs()
        
        NMR_file = get_path(config_data, nmr_type)
        print_to_console("NMR Plot: Run Plot_NMR")

        if nmr_type == "1H":
            data_dict = pd.read_csv(NMR_file) 

        elif nmr_type == "13C":
            data_dict = pd.read_csv(NMR_file)
        
        elif nmr_type == "HSQC":
            data_dict = pd.read_csv(NMR_file)
        
        elif nmr_type == "COSY":
            data_dict = pd.read_csv(NMR_file)    

        elif nmr_type == "IR":
            IR_data = {}
            NMR_file = get_path(config_data, "1H")
            data_dict = pd.read_csv(NMR_file)
            sample_ids = list(data_dict["sample-id"]) #keys())
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

        print_to_console("NMR Plot: Loading data successful")

        sample_ids = list(data_dict["sample-id"])
        if index < 0 or index >= len(sample_ids):
            return jsonify({"error": "Invalid molecule index"}), 404
        #import IPython; IPython.embed();

        current_sample_id = sample_ids[index]
        #import IPython; IPython.embed(); 

        # Get the shifts for the current sample ID
        nmr_data = data_dict.loc[data_dict['sample-id'] == current_sample_id, 'shifts'].values
        print_to_console("NMR Plot: Plot_NMR 3")

        if not nmr_data:
            return jsonify({"error": f"NMR data not available for {nmr_type}"}), 404
        try:
            if nmr_type == '13C':
                ppm = ast.literal_eval(nmr_data[0])
                intensity = [1] * len(ppm)
            elif nmr_type == 'HSQC':
                nmr_data = ast.literal_eval(nmr_data[0])
                ppm1, ppm2 = zip(*nmr_data)
                intensity = None
            elif nmr_type == 'COSY':
                nmr_data = ast.literal_eval(nmr_data[0])
                ppm1, ppm2 = zip(*nmr_data)
                intensity = None
            elif nmr_type == '1H':
                nmr_data = ast.literal_eval(nmr_data[0])
                ppm, intensity = zip(*nmr_data)
            elif nmr_type == 'IR':
                wave_lengths = IR_data.get('wave_lengths')
                absorbance = IR_data.get('absorbance')
            else:
                ppm = nmr_data
        except TypeError as e:
            print(f"Error extracting NMR data for {nmr_type}: {e}")
            return "Invalid NMR data format", 500
        print_to_console("NMR Plot: Extraction of datapoints successful")

        # Create Plotly figure
        fig = go.Figure()

        if nmr_type == 'COSY':
            fig.add_trace(go.Scatter(x=ppm1, y=ppm2, mode='markers', marker=dict(symbol='circle', size=8)))
            fig.update_layout(
                xaxis=dict(title='1H Chemical shift, ppm (δ)', autorange='reversed'),
                yaxis=dict(title='1H Chemical shift, ppm (δ)', autorange='reversed')
            )
        elif nmr_type == 'HSQC':
            fig.add_trace(go.Scatter(x=ppm1, y=ppm2, mode='markers', marker=dict(symbol='circle', size=8)))
            fig.update_layout(
                xaxis=dict(title='1H Chemical shift, ppm (δ)', autorange='reversed'),
                yaxis=dict(title='13C Chemical shift, ppm (δ)', autorange='reversed')
            )
        elif nmr_type == '1H':
            fig.add_trace(go.Scatter(x=ppm, y=[0] * len(ppm), mode='lines', name='Baseline', line=dict(color='blue')))
            for p, i in zip(ppm, intensity):
                fig.add_trace(go.Scatter(x=[p, p], y=[0, i], mode='lines', name='1H NMR', line=dict(color='blue'), hoverinfo='text', text=f'ppm: {p}, Intensity: {i}'))
            fig.update_layout(
                title='1H NMR Spectrum',
                xaxis=dict(title='1H Chemical shift, ppm (δ)', autorange='reversed'),
                yaxis=dict(title='Intensity', range=[-0.1, max(intensity)+0.2])
            )
        elif nmr_type == '13C':
            fig.add_trace(go.Scatter(x=ppm, y=[0] * len(ppm), mode='lines', name='Baseline', line=dict(color='green')))
            for p in ppm:
                fig.add_trace(go.Scatter(x=[p, p], y=[0, 1], mode='lines', name='13C NMR', line=dict(color='green'), hoverinfo='text', text=f'ppm: {p}'))
            fig.update_layout(
                title='13C NMR Spectrum',
                xaxis=dict(title='13C Chemical shift, ppm (δ)', autorange='reversed'),
                yaxis=dict(title='Intensity', range=[-0.1, 1.2])
            )
        elif nmr_type == 'IR':
            fig.add_trace(go.Scatter(x=wave_lengths, y=absorbance, mode='lines', name='IR Spectrum', line=dict(color='red')))
            fig.update_layout(
                title='IR Spectrum',
                xaxis=dict(title='Wavenumber (cm⁻¹)'),
                yaxis=dict(title='Absorbance')
            )


        # Convert Plotly figure to JSON and return
        graph_json = fig.to_json()
        print_to_console("NMR Plot: Plotting of Spectrum successful")

        return jsonify(graph_json)
    except Exception as e:
        logger.error(f"Error in plot_nmr: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/log_messages', methods=['POST'])
def log_messages():
    messages = request.json.get('messages', [])
    f.write_to_log_file(messages)
    return jsonify({"status": "Messages logged"})

@app.route('/test_model/<path:Checkpoint_Path>/<int:MNS_Value>/<string:spectral_types>')
def test_model(Checkpoint_Path, MNS_Value, spectral_types):
    try:
        print("start test_model")
        print_to_console("Testing model: Start test_model")

        greedy_full = False
        MW_filter = True
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        IR_config, config = load_configs()
        if Checkpoint_Path !="":
            config.checkpoint_path = "/" + Checkpoint_Path

        config.training_mode = spectral_types + "_MF_MW"
        config.multinom_runs = MNS_Value
        #config.data_size = 5
        print_to_console("Testing model: Configs loaded")

        model_MMT = mrtf.load_MMT_model(config)
        model_CLIP = mrtf.load_CLIP_model(config)
        print_to_console("Testing model: Model Loaded")

        val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
        val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")

        
        results_dict_bl_ZINC_ = mrtf.run_test_mns_performance_CLIP_3(config,  
                                                            model_MMT,
                                                            model_CLIP,
                                                            val_dataloader,
                                                            stoi, 
                                                            itos,
                                                            MW_filter)
        
        results_dict_bl_ZINC_, counter = mrtf.filter_invalid_inputs(results_dict_bl_ZINC_)
        
        avg_tani_bl_ZINC_, html_plot = rbgvm.plot_hist_of_results(results_dict_bl_ZINC_)
        

        print_to_console("Running test_model: Finished MNS Sampling")

        # Slow because also just takes one at the time
        if greedy_full == True:
            results_dict_greedy_bl_ZINC_, failed_bl_ZINC = mrtf.run_test_performance_CLIP_greedy_3(config,  
                                                                    stoi, 
                                                                    stoi_MF, 
                                                                    itos, 
                                                                    itos_MF)

            avg_tani_greedy_bl_ZINC_, html_plot_greedy = rbgvm.plot_hist_of_results_greedy(results_dict_greedy_bl_ZINC_)

        else: 
            config, results_dict_ZINC_greedy_bl_ = mrtf.run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi)
            avg_tani_greedy_bl_ZINC_ = results_dict_ZINC_greedy_bl_["tanimoto_mean"]
        
        total_results_bl_ZINC_ = mrtf.run_test_performance_CLIP_3(config, 
                                                            model_MMT, 
                                                            val_dataloader,
                                                            stoi)
        
        corr_sampleing_prob_bl_ZINC_ = total_results_bl_ZINC_["statistics_multiplication_avg"][0]
        print("avg_tani, avg_tani_greedy, corr_sampleing_prob'")
        print(avg_tani_bl_ZINC_, avg_tani_greedy_bl_ZINC_, corr_sampleing_prob_bl_ZINC_)       
        print("Greedy tanimoto results")      
        print_to_console("Running test_model: Finished Greedy Sampling")

        #######################################
        #### Just for testing load the data ###
        #######################################
        # file_path = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/Figures_Paper_2/precomputed_raw_data/20240701_IC_simACD_real_data/8.2_simACD_real_data_before_FT_MMT_v1.pkl"
        # with open(file_path, 'rb') as file:
        #         loaded_data = pickle.load(file)

        # # Extract the variables from the loaded data
        # results_dict_bl_ZINC_ = loaded_data.get('results_dict_bl_ZINC', False)
        # results_dict_ZINC_greedy_bl_ = loaded_data.get('results_dict_ZINC_greedy_bl', True)
        # avg_tani_bl_ZINC_ = loaded_data.get('avg_tani_bl_ZINC', True)
        # avg_tani_greedy_bl_ZINC_ = loaded_data.get('avg_tani_greedy_bl_ZINC', True)
        # total_results_bl_ZINC_ = loaded_data.get('total_results_bl_ZINC', True)
        # corr_sampleing_prob_bl_ZINC_ = loaded_data.get('corr_sampleing_prob_bl_ZINC', True)

        print_to_console("Running test_model: Variables extracted")

        img_io = plot_hist_of_results_combined(results_dict_ZINC_greedy_bl_, results_dict_bl_ZINC_)
        
        save_updated_config(config, config.config_path)
        
        print_to_console("Running test_model: Plotting finished")
        #import IPython; IPython.embed();

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"Error in test_model: {str(e)}")
        return jsonify({'error': str(e)}), 500


def test_model_IC():
    try:
        print("start test_model_IC")
        greedy_full = False
        MW_filter = True
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        IR_config, config = load_configs()
        #import IPython; IPython.embed();
        #config.data_size = 5
        print_to_console("Running test_model_IC: Configs loaded")

        model_MMT = mrtf.load_MMT_model(config)
        model_CLIP = mrtf.load_CLIP_model(config)

        val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
        val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")

        
        results_dict_bl_ZINC_ = mrtf.run_test_mns_performance_CLIP_3(config,  
                                                            model_MMT,
                                                            model_CLIP,
                                                            val_dataloader,
                                                            stoi, 
                                                            itos,
                                                            MW_filter)
        
        results_dict_bl_ZINC_, counter = mrtf.filter_invalid_inputs(results_dict_bl_ZINC_)
        
        avg_tani_bl_ZINC_, html_plot = rbgvm.plot_hist_of_results(results_dict_bl_ZINC_)
        
        # Slow because also just takes one at the time
        if greedy_full == True:
            results_dict_greedy_bl_ZINC_, failed_bl_ZINC = mrtf.run_test_performance_CLIP_greedy_3(config,  
                                                                    stoi, 
                                                                    stoi_MF, 
                                                                    itos, 
                                                                    itos_MF)

            avg_tani_greedy_bl_ZINC_, html_plot_greedy = rbgvm.plot_hist_of_results_greedy(results_dict_greedy_bl_ZINC_)

        else: 
            config, results_dict_ZINC_greedy_bl_ = mrtf.run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi)
            avg_tani_greedy_bl_ZINC_ = results_dict_ZINC_greedy_bl_["tanimoto_mean"]
        
        total_results_bl_ZINC_ = mrtf.run_test_performance_CLIP_3(config, 
                                                            model_MMT, 
                                                            val_dataloader,
                                                            stoi)
        
        corr_sampleing_prob_bl_ZINC_ = total_results_bl_ZINC_["statistics_multiplication_avg"][0]
        print("avg_tani, avg_tani_greedy, corr_sampleing_prob'")
        print(avg_tani_bl_ZINC_, avg_tani_greedy_bl_ZINC_, corr_sampleing_prob_bl_ZINC_)       
        print("Greedy tanimoto results")

        img_io = plot_hist_of_results_combined(results_dict_ZINC_greedy_bl_, results_dict_bl_ZINC_)      


        # #######################################
        # #### Just for testing load the data ###
        # #######################################
        # file_path = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/Figures_Paper_2/precomputed_raw_data/20240701_IC_simACD_real_data/8.2_simACD_real_data_before_FT_MMT_v1.pkl"
        # with open(file_path, 'rb') as file:
        #         loaded_data = pickle.load(file)

        # # Extract the variables from the loaded data
        # results_dict_bl_ZINC_ = loaded_data.get('results_dict_bl_ZINC', False)
        # results_dict_ZINC_greedy_bl_ = loaded_data.get('results_dict_ZINC_greedy_bl', True)
        # avg_tani_bl_ZINC_ = loaded_data.get('avg_tani_bl_ZINC', True)
        # avg_tani_greedy_bl_ZINC_ = loaded_data.get('avg_tani_greedy_bl_ZINC', True)
        # total_results_bl_ZINC_ = loaded_data.get('total_results_bl_ZINC', True)
        # corr_sampleing_prob_bl_ZINC_ = loaded_data.get('corr_sampleing_prob_bl_ZINC', True)

        print_to_console("Running test_model_IC: Variables extracted")
        
        #save_updated_config(config, config.config_path)
        
        print_to_console("Running test_model_IC: Finished IC test")
        #import IPython; IPython.embed();

        return corr_sampleing_prob_bl_ZINC_, img_io

    except Exception as e:
        print(f"Error in test_model_IC: {str(e)}")
        return jsonify({'error': str(e)}), 500


import pandas as pd

def select_samples_csv(file_path, data_size):
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Select the specified number of rows
    selected_rows = df.head(data_size)
    # Create the new file path with the _sel.csv suffix
    new_file_path = os.path.splitext(file_path)[0] + '_sel.csv'
    # Save the selected rows to the new file
    selected_rows.to_csv(new_file_path, index=False)
    print(f"Processed and saved {data_size} rows to {new_file_path}")
    # Return the path of the new file
    return new_file_path


@app.route('/run_IC/<path:model_save_dir>/<int:MF_generations>/<int:num_epochs>/<int:MF_delta_weight>/<int:max_scaffold_generations>/<float:lr_pretraining>/<float:IC_threshold>/<int:data_size>')
def run_IC(model_save_dir, MF_generations, num_epochs, MF_delta_weight, max_scaffold_generations, lr_pretraining, IC_threshold, data_size):   

    itos, stoi, stoi_MF, itos_MF = load_json_dics()
    IR_config, config = load_configs()
    config.data_size = data_size
    # Save the selected smiles to the right location for MF to work
    config.SGNN_csv_gen_smi  = ex.filter_invalid_criteria(config.SGNN_csv_gen_smi)
    df = pd.read_csv(config.SGNN_csv_gen_smi)
    # Save DataFrame to CSV
    script_dir = os.path.dirname(__file__)
    base_path = os.path.abspath(os.path.join(script_dir, '../../deep-molecular-optimization/data/MMP'))
    df = df.iloc[:config.data_size]
    data_size_backup = config.data_size
    csv_file_path = os.path.join(base_path, 'test_selection_2.csv')

    #csv_file_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/data/MMP/test_selection_2.csv'
    df.to_csv(csv_file_path, index=False)
    print_to_console("Running run_IC: Loaded Config")


    # Create a new folder with current date and time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_name = f"run_{current_time}"
    new_folder_path = os.path.join("/"+model_save_dir, new_folder_name)
    
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Created new directory: {new_folder_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return jsonify({"error": str(e)}), 500

    config.model_save_dir = new_folder_path
    config.MF_generations = MF_generations
    config.num_epochs = num_epochs
    config.MF_delta_weight = MF_delta_weight
    config.max_scaffold_generations = max_scaffold_generations
    config.IC_threshold = IC_threshold
    config.lr_pretraining = lr_pretraining

    performance = 0
    print_to_console("Running run_IC: Starting Improvement Cycle")
    while True:
        config.execution_type = "SMI_generation_MF"
        if config.execution_type == "SMI_generation_MF":

            #config.n_samples = config.data_size

            print("\033[1m\033[31mThis is: SMI_generation_MF\033[0m")
            config, results_dict_MF = ex.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)

            # Iterate through the dictionary and remove 'nan' from lists
            results_dict_MF = {key: value for key, value in results_dict_MF.items() if not hf.contains_only_nan(value)}
            for key, value in results_dict_MF.items():
                results_dict_MF[key] = hf.remove_nan_from_list(value)

            #combined_list_MF, html_TSNE, html_UMAP, html_PCA = cv.plot_cluster_MF(results_dict_MF, config)
            transformed_list_MF = [[key, value] for key, value in results_dict_MF.items()]
            src_smi_MF, key_len_MF = list(results_dict_MF.keys()), len(list(results_dict_MF.keys()))

            combined_list_MF = [item for sublist in transformed_list_MF for item in sublist[1][:]]
            #number_point_list_MF = [len(sublist[1]) for sublist in transformed_list_MF ]
            #combined_list_new_MF = src_smi_MF + combined_list_MF

            config.execution_type = "combine_MMT_MF"
            print(config.data_size)
        print("END SMI_generation_MF")

        ########################################################################
        #config.execution_type = "combine_MMT_MF"
        combined_list_MMT = []  # in case if I want to use MMT for generating molecules for training
        if config.execution_type == "combine_MMT_MF":
            print("\033[1m\033[31mThis is: combine_MMT_MF\033[0m")
            all_gen_smis = combined_list_MMT + combined_list_MF
            all_gen_smis = [smiles for smiles in all_gen_smis if smiles != 'NAN']

            #filter out potential hits from the real test_set
            val_data = pd.read_csv(config.csv_path_val)
            all_gen_smis = mrtf.filter_smiles(val_data, all_gen_smis)
            length_of_list = len(all_gen_smis)   
            random_number_strings = [f"GT_{str(i).zfill(7)}" for i in range(1, length_of_list + 1)]
            aug_mol_df = pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})
            config.execution_type = "blend_prev_train_data"
        print("END combine_MMT_MF")

        ########################################################################
        config.train_data_blend = 0
        config.execution_type = "blend_prev_train_data"
        if config.execution_type == "blend_prev_train_data":
            print("\033[1m\033[31mThis is: blend_prev_train_data\033[0m")
            config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)
            config.execution_type = "data_generation"
        print("END blend_prev_train_data")

        ########################################################################
        config.execution_type = "data_generation"
        if config.execution_type == "data_generation":
            #config.csv_SMI_targets = config.csv_1H_path_SGNN
            print("\033[1m\033[31mThis is: data_generation\033[0m")
            config = ex.gen_sim_aug_data(config, IR_config)
            config.execution_type = "transformer_improvement"
            sim_data_gen = True
        print("END data_generation")

        ########################################################################
        #config.execution_type = "transformer_improvement"
        #sim_data_gen = True
        if config.execution_type == "transformer_improvement" and sim_data_gen == True:
            print("\033[1m\033[31mThis is: transformer_improvement, sim_data_gen == TRUE\033[0m")
            config.training_setup = "pretraining"
            mtf.run_MMT(config, stoi, stoi_MF)
        print("END transformer_improvement")

        ########################################################################
        config = ex.update_model_path(config)

        config.csv_1H_path_SGNN = config.csv_1H_path_display 
        config.csv_13C_path_SGNN = config.csv_13C_path_display 
        config.csv_HSQC_path_SGNN = config.csv_HSQC_path_display 
        config.csv_COSY_path_SGNN = config.csv_COSY_path_display 
        config.IR_data_folder = config.IR_data_folder_display 

        #import IPython; IPython.embed();
        config.data_size = data_size_backup
        # Process each CSV file according to the paths in the config
        config.csv_path_val = select_samples_csv(config.csv_1H_path_SGNN, config.data_size)
        config.pickle_file_path = ""
        save_updated_config(config, config.config_path)

        #### TODO MARTIN add a new second test_model function
        performance, img_io = test_model_IC()
        print("performance")
        print(performance)
        print_to_console("Performance: " + str(performance))

        if performance > IC_threshold:
            break
        

    #######################################
    #### Just for testing load the data ###
    #######################################
    # file_path = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/Figures_Paper_2/precomputed_raw_data/20240701_IC_simACD_real_data/8.2_simACD_real_data_before_FT_MMT_v1.pkl"
    # with open(file_path, 'rb') as file:
    #         loaded_data = pickle.load(file)

    # # Extract the variables from the loaded data
    # results_dict_bl_ZINC_ = loaded_data.get('results_dict_bl_ZINC', False)
    # results_dict_ZINC_greedy_bl_ = loaded_data.get('results_dict_ZINC_greedy_bl', True)
    # avg_tani_bl_ZINC_ = loaded_data.get('avg_tani_bl_ZINC', True)
    # avg_tani_greedy_bl_ZINC_ = loaded_data.get('avg_tani_greedy_bl_ZINC', True)
    # total_results_bl_ZINC_ = loaded_data.get('total_results_bl_ZINC', True)
    # corr_sampleing_prob_bl_ZINC_ = loaded_data.get('corr_sampleing_prob_bl_ZINC', True)
    #img_io = plot_hist_of_results_combined(results_dict_ZINC_greedy_bl_, results_dict_bl_ZINC_)
    print_to_console("Running run_IC: Finished")

    save_updated_config(config, config.config_path)

    return send_file(img_io, mimetype='image/png')



def sort_and_rank_chemical_data(data):
    sorted_data = {}
    
    for smiles, compounds_list in data.items():
        compounds = compounds_list[0]  # Access the list of compounds
        df = compounds_list[1]  # Access the dataframe

        rankings = [compound[-1] for compound in compounds]  # Last item of each compound
        
        # Separate the two float values
        rankings_1 = [item[0] for item in rankings]
        rankings_2 = [item[1] for item in rankings]
        
        # Create rankings for each float value
        rank_1 = np.argsort(np.argsort(rankings_1))
        rank_2 = np.argsort(np.argsort(rankings_2))
        
        # Combine rankings (you can adjust the weights if needed)
        combined_ranks = (rank_1 + rank_2) / 2
        
        # Add combined rank to each compound
        for i, compound in enumerate(compounds):
            compound.append(combined_ranks[i])
        
        # Sort the compounds based on the new ranking
        sorted_compounds = sorted(compounds, key=lambda x: x[-1])
        
        # Store the sorted compounds back in the dictionary
        sorted_data[smiles] = [sorted_compounds, df]
        
    return sorted_data

  

@app.route('/run_model_exp_data')
def run_model_exp_data():
    print_to_console("Running run_model_exp_data: Start")

    Checkpoint_Path = urllib.parse.unquote(request.args.get('Checkpoint_Path'))
    MNS_Value = int(request.args.get('MNS_Value'))
    REAL_1H_Path = urllib.parse.unquote(request.args.get('REAL_1H_Path'))
    REAL_13C_Path = urllib.parse.unquote(request.args.get('REAL_13C_Path'))
    REAL_HSQC_Path = urllib.parse.unquote(request.args.get('REAL_HSQC_Path'))
    REAL_COSY_Path = urllib.parse.unquote(request.args.get('REAL_COSY_Path'))
    REAL_IR_Path = urllib.parse.unquote(request.args.get('REAL_IR_Path'))
    DataSize = int(request.args.get('DataSize'))
    # import IPython; IPython.embed();
 
    try:
        print("start run_model_exp_data")
        greedy_full = False
        MW_filter = True
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        IR_config, config = load_configs()

        if Checkpoint_Path != "":
            config.checkpoint_path = Checkpoint_Path
            
        config.csv_1H_path_REAL = REAL_1H_Path
        config.csv_13C_path_REAL = REAL_13C_Path
        config.csv_HSQC_path_REAL = REAL_HSQC_Path
        config.csv_COSY_path_REAL = REAL_COSY_Path
        config.IR_path_REAL = REAL_IR_Path

        config.csv_path_val = config.csv_1H_path_REAL
        config.csv_1H_path_SGNN = config.csv_1H_path_REAL
        config.csv_13C_path_SGNN = config.csv_13C_path_REAL
        config.csv_HSQC_path_SGNN = config.csv_HSQC_path_REAL
        config.csv_COSY_path_SGNN = config.csv_COSY_path_REAL
        config.IR_data_folder = config.IR_path_REAL

        config.pickle_file_path = ""
        config.multinom_runs = MNS_Value
        config.data_size = DataSize

        config.csv_path_val = select_samples_csv(config.csv_1H_path_REAL, config.data_size)

        save_updated_config(config, config.config_path)

        print_to_console("Running model: Configs loaded")

        model_MMT = mrtf.load_MMT_model(config)
        model_CLIP = mrtf.load_CLIP_model(config)

        val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
        val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")

        
        results_dict_bl_ZINC_ = mrtf.run_test_mns_performance_CLIP_3(config,  
                                                            model_MMT,
                                                            model_CLIP,
                                                            val_dataloader,
                                                            stoi, 
                                                            itos,
                                                            MW_filter)
        
        results_dict_bl_ZINC_, counter = mrtf.filter_invalid_inputs(results_dict_bl_ZINC_)
        
        avg_tani_bl_ZINC_, html_plot = rbgvm.plot_hist_of_results(results_dict_bl_ZINC_)

        # Slow because also just takes one at the time
        if greedy_full == True:
            results_dict_greedy_bl_ZINC_, failed_bl_ZINC = mrtf.run_test_performance_CLIP_greedy_3(config,  
                                                                    stoi, 
                                                                    stoi_MF, 
                                                                    itos, 
                                                                    itos_MF)

            avg_tani_greedy_bl_ZINC_, html_plot_greedy = rbgvm.plot_hist_of_results_greedy(results_dict_greedy_bl_ZINC_)

        else: 
            config, results_dict_ZINC_greedy_bl_ = mrtf.run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi)
            avg_tani_greedy_bl_ZINC_ = results_dict_ZINC_greedy_bl_["tanimoto_mean"]
        
        total_results_bl_ZINC_ = mrtf.run_test_performance_CLIP_3(config, 
                                                            model_MMT, 
                                                            val_dataloader,
                                                            stoi)
        
        corr_sampleing_prob_bl_ZINC_ = total_results_bl_ZINC_["statistics_multiplication_avg"][0]
        print("avg_tani, avg_tani_greedy, corr_sampleing_prob'")
        print(avg_tani_bl_ZINC_, avg_tani_greedy_bl_ZINC_, corr_sampleing_prob_bl_ZINC_)       
        print("Greedy tanimoto results")

        # #######################################
        # #### Just for testing load the data ###
        # #######################################
        # file_path = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/Figures_Paper_2/precomputed_raw_data/20240701_IC_simACD_real_data/8.2_simACD_real_data_before_FT_MMT_v1.pkl"
        # with open(file_path, 'rb') as file:
        #         loaded_data = pickle.load(file)

        # # Extract the variables from the loaded data
        # results_dict_bl_ZINC_ = loaded_data.get('results_dict_bl_ZINC', False)
        # results_dict_ZINC_greedy_bl_ = loaded_data.get('results_dict_ZINC_greedy_bl', True)
        # avg_tani_bl_ZINC_ = loaded_data.get('avg_tani_bl_ZINC', True)
        # avg_tani_greedy_bl_ZINC_ = loaded_data.get('avg_tani_greedy_bl_ZINC', True)
        # total_results_bl_ZINC_ = loaded_data.get('total_results_bl_ZINC', True)
        # corr_sampleing_prob_bl_ZINC_ = loaded_data.get('corr_sampleing_prob_bl_ZINC', True)
    
        img_io = plot_hist_of_results_combined(results_dict_ZINC_greedy_bl_, results_dict_bl_ZINC_)
   
        results_dict_bl_ZINC_ = sort_and_rank_chemical_data(results_dict_bl_ZINC_)

        # Define the variables to save based on the results of your experiment
        variables_to_save = {
            #'prob_dict_results_1c': prob_dict_results_1c_,
            #'results_dict_1c': results_dict_1c_,
            'results_dict_bl_ZINC': results_dict_bl_ZINC_,
            'avg_tani_bl_ZINC': avg_tani_bl_ZINC_,
            'results_dict_greedy_bl_ZINC': results_dict_greedy_bl_ZINC_ if greedy_full else None,
            'failed_bl_ZINC': failed_bl_ZINC if greedy_full else None,
            'avg_tani_greedy_bl_ZINC': avg_tani_greedy_bl_ZINC_ if greedy_full else None,
            'results_dict_ZINC_greedy_bl': results_dict_ZINC_greedy_bl_ if not greedy_full else None,
            'total_results_bl_ZINC': total_results_bl_ZINC_,
            'corr_sampleing_prob_bl_ZINC': corr_sampleing_prob_bl_ZINC_,
        }

     
        # Generate filename with date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{current_time}_IC_exp_data.pkl"

        # Construct the full path
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)

        # Define base path relative to the script's location
        base_path = os.path.abspath(os.path.join(script_dir, '..', 'pkl_files'))

        # Define the path to the directory where files should be saved
        save_dir = base_path

        #save_dir = "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/pkl_files"
        full_path = os.path.join(save_dir, filename)
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the pickle file
        with open(full_path, 'wb') as f:
            pickle.dump(variables_to_save, f)

        # Save the path to the config
        config.exp_pkl_path = full_path

        print_to_console(f"Running run_model_exp_data: Pickle file saved at: {full_path}")

        # Save the updated config
        save_updated_config(config, config.config_path)

        print_to_console("Running run_model_exp_data: Finished")

        # #######################################
        # #### Just for testing load the data ###
        # #######################################
        # file_path = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/___FIGURES_PAPERS/Figures_Paper_2/precomputed_raw_data/20240701_IC_simACD_real_data/8.2_simACD_real_data_before_FT_MMT_v1.pkl"
        # with open(file_path, 'rb') as file:
        #         loaded_data = pickle.load(file)

        # # Extract the variables from the loaded data
        # results_dict_bl_ZINC_ = loaded_data.get('results_dict_bl_ZINC', False)
        # results_dict_ZINC_greedy_bl_ = loaded_data.get('results_dict_ZINC_greedy_bl', True)
        # avg_tani_bl_ZINC_ = loaded_data.get('avg_tani_bl_ZINC', True)
        # avg_tani_greedy_bl_ZINC_ = loaded_data.get('avg_tani_greedy_bl_ZINC', True)
        # total_results_bl_ZINC_ = loaded_data.get('total_results_bl_ZINC', True)
        # corr_sampleing_prob_bl_ZINC_ = loaded_data.get('corr_sampleing_prob_bl_ZINC', True)
        print("end run_model_exp_data")

        return img_io

    except Exception as e:
        print(f"Error in run_model_exp_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/colored_molecule/<int:index>/<int:i>')
def colored_molecule(index, i):
    try:
        colors = []
        label = ""
        IR_config, config = load_configs()
        with open(config.exp_pkl_path, 'rb') as file:
        #with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/pkl_files/20240716_053237_IC_exp_data.pkl', 'rb') as file:
            data = pickle.load(file)

        result = data['results_dict_bl_ZINC']
        keys_list = list(result.keys())
        target_smi = keys_list[index]

        if i == 0:
            # Process target SMILES
            smi = target_smi
            prob_list = [1 for _ in smi]
            label = "Target"
        else:
            # Process generated SMILES
            smi = result[target_smi][0][i-1][0]
            prob_list = result[target_smi][0][i-1][3].tolist()
            HSQC_err = result[target_smi][0][i-1][-2][0]
            COSY_err = result[target_smi][0][i-1][-2][1]
            tani_sim = result[target_smi][0][i-1][-3]
            prob_product = reduce(operator.mul, prob_list, 1)
            label += f"Generation: {i}\n Probability: {prob_product:.2e}\n HSQC Error: {HSQC_err:.3f} \n COSY Error: {COSY_err:.3f} \nTanimoto Sim: {tani_sim:.3f}" 

        svg = f.generate_colored_molecule(smi, prob_list, label=label)

        #Generate the colored SMILE
        html_string = f.generate_colored_html(smi, prob_list)
        
        # Return JSON instead of raw SVG
        return jsonify({
            'colored_molecule': svg,
            'colored_smile': html_string,
            'smile': smi
        })

    except Exception as e:
        print("Error in colored_molecule:", str(e))  # Add this line
        return jsonify({'error': str(e)}), 400




def process_sdf_file(sdf_file_path, config, IR_config, trash_folder="trash_folder"):
    # Create trash folder if it doesn't exist
    trash_folder_path = Path(trash_folder)

    # Delete the trash folder if it exists
    if trash_folder_path.exists():
        shutil.rmtree(trash_folder_path)

    trash_folder_path.mkdir(parents=True, exist_ok=True)

    # Copy SDF file to trash folder
    sdf_filename = os.path.basename(sdf_file_path)
    destination_path = trash_folder_path / sdf_filename
    shutil.copy2(sdf_file_path, destination_path)

    # Update config
    config.SGNN_gen_folder_path = str(trash_folder_path)

    # Run NMR generation functions
    data_1H, csv_1H_path = dl.run_1H_generation(config)
    print("\033[1m\033[33m run_1H_generation: DONE\033[0m")

    data_13C, csv_13C_path = dl.run_13C_generation(config)
    print("\033[1m\033[33m run_13C_generation: DONE\033[0m")

    data_COSY, csv_COSY_path = dl.run_COSY_generation(config)
    print("\033[1m\033[33m run_COSY_generation: DONE\033[0m")

    data_HSQC, csv_HSQC_path = dl.run_HSQC_generation(config)
    print("\033[1m\033[33m run_HSQC_generation: DONE\033[0m")
    
    config.csv_SMI_targets = csv_1H_path
    config.SGNN_gen_folder_path = os.path.dirname(csv_1H_path)
    data_IR = irs.run_IR_simulation(config, IR_config, "target")


    # Compile results
    nmr_data = {
        '1H': data_1H,
        '13C': data_13C,
        'COSY': data_COSY,
        'HSQC': data_HSQC
    }

    csv_paths = {
        '1H': csv_1H_path,
        '13C': csv_13C_path,
        'COSY': csv_COSY_path,
        'HSQC': csv_HSQC_path,
        "IR": data_IR
    }

    return nmr_data, csv_paths


@app.route('/plot_dual_NMR')
def plot_dual_NMR():
    try:
        print_to_console("Running plot_dual_NMR: Start")
        IR_config, config = load_configs()
        #import IPython; IPython.embed();
        nmr_type = request.args.get('nmr_type', session.get('nmr_type', '1H'))
        exp_index = int(request.args.get('exp_index', 0))
        sim_index = int(request.args.get('sim_index', 0))
        session['nmr_type'] = nmr_type

        # Load sample IDs from 1H data (assuming 1H data contains all sample IDs)
        sample_ids_df = pd.read_csv(config.csv_1H_path_REAL)
        sample_ids = list(sample_ids_df["sample-id"])
        if exp_index < 0 or exp_index >= len(sample_ids):
            return jsonify({"error": "Invalid molecule index"}), 404
        
        current_sample_id = sample_ids[exp_index]

        # Load Experimental data
        if nmr_type == "1H":
            REAL_data_dict = pd.read_csv(config.csv_1H_path_REAL)
        elif nmr_type == "13C":
            REAL_data_dict = pd.read_csv(config.csv_13C_path_REAL)
        elif nmr_type == "HSQC":
            REAL_data_dict = pd.read_csv(config.csv_HSQC_path_REAL)
        elif nmr_type == "COSY":
            REAL_data_dict = pd.read_csv(config.csv_COSY_path_REAL)
        elif nmr_type == "IR":
            IR_csv_path = os.path.join(config.IR_path_REAL, f"{current_sample_id}.csv")
            IR_df = pd.read_csv(IR_csv_path)

            if IR_df.shape[1] != 1:
                logger.error(f"Invalid column count for IR data in {IR_csv_path}")
                return jsonify({"error": "Invalid IR data format"}), 400

            absorbance = IR_df.iloc[:, 0].astype(float).tolist()
            wave_numbers = np.linspace(400, 4000, len(absorbance))
            IR_exp_data = {'wave_lengths': wave_numbers, 'absorbance': absorbance}
            exp_data = sample_ids_df[sample_ids_df['sample-id'] == current_sample_id].iloc[0]
            exp_smi = exp_data["SMILES"]            

        print_to_console("Plot_dual_NMR: Experimental data loaded")

        # Extract experimental data for the current sample
        if nmr_type != "IR":
            exp_data = REAL_data_dict[REAL_data_dict['sample-id'] == current_sample_id].iloc[0]
            exp_smi = exp_data["SMILES"]
            exp_shifts_str = exp_data["shifts"]
            exp_data_list = ast.literal_eval(exp_shifts_str)

        # Load Simulated Data
        with open(config.exp_pkl_path, 'rb') as file:
            data = pickle.load(file)
        result = data['results_dict_bl_ZINC']
        
        # Ensure the exp_smi matches the one in the results
        if exp_smi not in result:
            return jsonify({"error": "Experimental SMILES not found in simulation results"}), 404

        gen_smi = result[exp_smi][0][sim_index][0]
        
        df = result[exp_smi][-1]
        matching_row = df[df['SMILES'] == gen_smi]

        #trash_folder = '/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/dump/trash'
        sdf_file_path = matching_row["sdf_path"].item()
        nmr_data, csv_paths = process_sdf_file(sdf_file_path, config, IR_config)

        if nmr_type in ["1H", "13C", "HSQC", "COSY"]:
            SIM_data_dict = pd.read_csv(csv_paths[nmr_type])
            shifts_str = SIM_data_dict["shifts"].iloc[0]
            sim_data_list = ast.literal_eval(shifts_str)
        elif nmr_type == "IR":
            IR_sim_folder = csv_paths["IR"]
            data_dict = pd.read_csv(csv_paths["1H"])
            sample_id = list(data_dict["sample-id"])[0]
            IR_sim_csv_path = os.path.join(IR_sim_folder, f"{sample_id}.csv")
            IR_sim_df = pd.read_csv(IR_sim_csv_path)

            if IR_sim_df.shape[1] != 1:
                logger.error(f"Invalid column count for IR data in {IR_sim_csv_path}")
                return jsonify({"error": "Invalid IR data format"}), 400

            absorbance = IR_sim_df.iloc[:, 0].astype(float).tolist()
            wave_numbers = np.linspace(400, 4000, len(absorbance))
            IR_sim_data = {'wave_lengths': wave_numbers, 'absorbance': absorbance}

        # Create Plotly figure
        fig = go.Figure()

        # Plot spectra based on type
        if nmr_type in ['1H', '13C']:
            ppm1, intensity1 = zip(*exp_data_list) if nmr_type == '1H' else (exp_data_list, [1] * len(exp_data_list))
            ppm2, intensity2 = zip(*sim_data_list) if nmr_type == '1H' else (sim_data_list, [1] * len(sim_data_list))

            fig.add_trace(go.Scatter(x=ppm1, y=[0] * len(ppm1), mode='lines', name='Exp Baseline', line=dict(color='blue')))
            for p, i in zip(ppm1, intensity1):
                fig.add_trace(go.Scatter(x=[p, p], y=[0, i], mode='lines', name=f'Exp {nmr_type}', line=dict(color='blue'), hoverinfo='text', text=f'ppm: {p}, Intensity: {i}'))

            fig.add_trace(go.Scatter(x=ppm2, y=[0] * len(ppm2), mode='lines', name='Sim Baseline', line=dict(color='red')))
            for p, i in zip(ppm2, intensity2):
                fig.add_trace(go.Scatter(x=[p, p], y=[0, i], mode='lines', name=f'Sim {nmr_type}', line=dict(color='red'), hoverinfo='text', text=f'ppm: {p}, Intensity: {i}'))

            fig.update_layout(
                title=f'Dual {nmr_type} NMR Spectrum',
                xaxis=dict(title=f'{nmr_type} Chemical shift, ppm (δ)', autorange='reversed'),
                yaxis=dict(title='Intensity', range=[-0.1, max(max(intensity1), max(intensity2)) + 0.2])
            )

        elif nmr_type in ['HSQC', 'COSY']:
            ppm1_1, ppm2_1 = zip(*exp_data_list)
            ppm1_2, ppm2_2 = zip(*sim_data_list)

            fig.add_trace(go.Scatter(x=ppm1_1, y=ppm2_1, mode='markers', name=f'Exp {nmr_type}', marker=dict(color='blue', symbol='circle', size=8)))
            fig.add_trace(go.Scatter(x=ppm1_2, y=ppm2_2, mode='markers', name=f'Sim {nmr_type}', marker=dict(color='red', symbol='circle', size=8)))

            x_title = '1H Chemical shift, ppm (δ)'
            y_title = '13C Chemical shift, ppm (δ)' if nmr_type == 'HSQC' else '1H Chemical shift, ppm (δ)'
            fig.update_layout(
                title=f'Dual {nmr_type} Spectrum',
                xaxis=dict(title=x_title, autorange='reversed'),
                yaxis=dict(title=y_title, autorange='reversed')
            )

        elif nmr_type == 'IR':
            wave_lengths1, absorbance1 = IR_exp_data['wave_lengths'], IR_exp_data['absorbance']
            wave_lengths2, absorbance2 = IR_sim_data['wave_lengths'], IR_sim_data['absorbance']

            fig.add_trace(go.Scatter(x=wave_lengths1, y=absorbance1, mode='lines', name='Exp IR Spectrum', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=wave_lengths2, y=absorbance2, mode='lines', name='Sim IR Spectrum', line=dict(color='red')))

            fig.update_layout(
                title='Dual IR Spectrum',
                xaxis=dict(title='Wavenumber (cm⁻¹)'),
                yaxis=dict(title='Absorbance')
            )

        # Convert Plotly figure to JSON and return
        graph_json = fig.to_json()
        print_to_console("Running plot_dual_NMR: Finished")

        return jsonify(graph_json)

    except Exception as e:
        logger.error(f"Error in plot_dual_spectra: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def print_to_console(message):
    print(f"Emitting message: {message}")  # Debug print
    socketio.emit('console_message', {'message': message})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', debug=True, port=8083)
   # app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug to False in production

