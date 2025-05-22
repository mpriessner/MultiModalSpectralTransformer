from flask import Flask, render_template, send_file, request, redirect, flash, jsonify, session
from flask_socketio import SocketIO, emit
import os
import json
from rdkit import Chem
from rdkit.Chem import Draw
import io
import csv
import ast
import random
import numpy as np
from rdkit.Chem import AllChem
from umap import UMAP
from tqdm import tqdm
import plotly.graph_objects as go
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import glob
import logging
from datetime import datetime
import pickle
import pandas as pd


molecules = {}
#config_PATH = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/config.json'
# Define the base path relative to the script location
script_dir = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(script_dir, ''))

# Update paths relative to the script's location
config_PATH = os.path.join(base_path, 'config.json')
log_folder = os.path.join(base_path, 'Log_Folder')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

        
# def write_to_log_file(messages):
#     config_file_path = config_PATH
#     # Read the existing config file if it exists
#     if os.path.exists(config_file_path):
#         with open(config_file_path, 'r') as json_file:
#             config_data = json.load(json_file)
#     else:
#         config_data = {}

#     log_folder = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/Log_Folder'  # Name of the folder to store logs
#     if not os.path.exists(log_folder):
#         os.makedirs(log_folder)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file_name = f'console_log_{timestamp}.txt'
#     log_file_path = os.path.join(log_folder, log_file_name)
    
#     with open(log_file_path, 'w') as log_file:
#         for message in messages:
#             log_file.write(f"{datetime.now()}: {message}\n")

#         data = {}
#         data['Console_Log_File'] = log_file_path

#     # Update the config data with the new SMILES data
#     config_data.update(data)

#     # Write the updated config data back to the JSON file
#     with open(config_file_path, 'w') as json_file:
#         json.dump(config_data, json_file, indent=4)    


def write_to_log_file(messages):
    # Read the existing config file if it exists
    if os.path.exists(config_PATH):
        with open(config_PATH, 'r') as json_file:
            config_data = json.load(json_file)
    else:
        config_data = {}

    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    # Define log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'console_log_{timestamp}.txt'
    log_file_path = os.path.join(log_folder, log_file_name)
    
    # Write messages to log file
    with open(log_file_path, 'w') as log_file:
        for message in messages:
            log_file.write(f"{datetime.now()}: {message}\n")

        data = {}
        data['Console_Log_File'] = log_file_path

    # Update the config data with the new log file path
    config_data.update(data)

    # Write the updated config data back to the JSON file
    with open(config_PATH, 'w') as json_file:
        json.dump(config_data, json_file, indent=4)

def process_probabilities(smile, probabilities):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    num_atoms = mol.GetNumAtoms()
    atom_probabilities = []
    current_atom_prob = None
    non_atom_probs = []

    for char, prob in zip(smile, probabilities):
        if char.isalpha() or char == '[' or char == ']':  # Start of an atom symbol
            if current_atom_prob is not None:
                if non_atom_probs:
                    # Average the non-atom probabilities with the current and previous atom
                    avg_prob = (current_atom_prob + sum(non_atom_probs) + prob) / (len(non_atom_probs) + 2)
                    atom_probabilities.append(avg_prob)
                    non_atom_probs = []
                else:
                    atom_probabilities.append(current_atom_prob)
            current_atom_prob = prob
        else:  # Non-atom character
            non_atom_probs.append(prob)

    # Handle the last atom
    if current_atom_prob is not None:
        if non_atom_probs:
            # Average the non-atom probabilities with the last atom
            avg_prob = (current_atom_prob + sum(non_atom_probs)) / (len(non_atom_probs) + 1)
            atom_probabilities.append(avg_prob)
        else:
            atom_probabilities.append(current_atom_prob)

    # Ensure we have the correct number of probabilities
    if len(atom_probabilities) > num_atoms:
        atom_probabilities = atom_probabilities[:num_atoms]
    if len(atom_probabilities) != num_atoms:
        warnings.warn(f"Number of processed probabilities ({len(atom_probabilities)}) "
                      f"does not match number of atoms in molecule ({num_atoms}). "
                      "This may lead to incorrect coloring."
                      "For now probabilities of 1 are added to keep the code working")
        atom_probabilities.extend([1.0] * (num_atoms - len(atom_probabilities)))

    return atom_probabilities

def generate_colored_molecule(smile, probabilities, label=None):
    molecule = Chem.MolFromSmiles(smile)
    atom_probabilities = process_probabilities(smile, probabilities)
    
    d = rdMolDraw2D.MolDraw2DSVG(300, 300)
    d.drawOptions().useBWAtomPalette()
    
    highlight_atom_colors = {i: get_color(prob) for i, prob in enumerate(atom_probabilities)}
    
    d.DrawMolecule(molecule, highlightAtoms=list(highlight_atom_colors.keys()), highlightAtomColors=highlight_atom_colors)
    d.FinishDrawing()
    
    svg = d.GetDrawingText().replace('\n', '')
    
    if label:
        lines = label.split('\n')
        label_svg = '<text x="10" y="20" font-family="sans-serif" font-size="14px" fill="black">'
        for i, line in enumerate(lines):
            label_svg += f'<tspan x="10" dy="{20 if i == 0 else 20}">{line}</tspan>'
        label_svg += '</text>'
        svg = svg.replace('</svg>', f'{label_svg}</svg>')
    return svg

def generate_colored_html(smile, probabilities):
    html_string = '<span style="font-family: monospace;">'
    
    for char, prob in zip(smile, probabilities):
        color = get_color(prob)
        css_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
        html_string += f'<span style="background-color: {css_color}; padding: 1px; display: inline-block;">{char}</span>'
    
    html_string += '</span>'
    return html_string

def get_color(prob):
    green_value = prob * 0.9
    red_value = 1.0 - prob * 0.40
    blue_value = 0.7
    return (red_value, green_value, blue_value)




def generate_morgan_fingerprints():
    fingerprints = []
    smiles_list = []
    for smiles in molecules.keys():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=512)
            fingerprints.append(fingerprint)
            smiles_list.append(smiles)
    return fingerprints, smiles_list

def perform_umap(fingerprints):
    umap_model = UMAP(n_components=2)
    transformed_fingerprints = umap_model.fit_transform(fingerprints)
    return transformed_fingerprints


def parse_NMR_csv(file_path, nmr_type):
    global IR_data
    if nmr_type is None:
        logger.error(f"Unknown NMR type in file path: {file_path}")
        return

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)
    except IOError as e:
        logger.error(f"Error opening file {file_path}: {e}")
        return
    except csv.Error as e:
        logger.error(f"CSV error in file {file_path}: {e}")
        return
    
    absorbance = []

    for row in data:
        try:
            if nmr_type == 'IR':
                if len(row) != 1:
                    logger.warning(f"Skipping invalid row in {file_path}: {row}")
                    continue
                absorbance.append(float(row[0]))
                wave_numbers = np.linspace(400, 4000, len(data))
                wave_lengths = 1 / wave_numbers * 10**4  # in micrometers
                IR_data = {'wave_lengths': wave_lengths, 'absorbance': absorbance}
            else:
                if len(row) < 2:
                    logger.warning(f"Skipping invalid row in {file_path}: {row}")
                    continue

                smiles, raw_data, zinc_id = row if len(row) == 3 else (row[0], ",".join(row[1:-1]), row[-1])
                
                smiles = smiles.strip('\"')
                raw_data = raw_data.strip('\"')
                zinc_id = zinc_id.strip('\"')

                # Validate SMILES
                if not Chem.MolFromSmiles(smiles):
                    logger.warning(f"Invalid SMILES in {file_path}: {smiles}")
                    continue

                # Parse NMR data based on type
                try:
                    if nmr_type == '13C':
                        data = ast.literal_eval(raw_data)
                    elif nmr_type == 'HSQC':
                        data = [[float(pair[0]), float(pair[1])] for pair in ast.literal_eval(raw_data)]
                    elif nmr_type == 'COSY':
                        data = [(float(pair[0]), float(pair[1])) for pair in ast.literal_eval(raw_data)]
                    else:
                        data = ast.literal_eval(raw_data)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing NMR data in {file_path} for {zinc_id}: {e}")
                    continue

                if zinc_id not in molecules:
                    molecules[zinc_id] = {
                        'smiles': smiles,
                        'zinc_id': zinc_id,
                        'molecular_weight': round(random.uniform(100, 500), 2),
                        'probability': round(random.uniform(0, 100), 2),
                        'nmr_data': {}
                    }
                molecules[zinc_id]['nmr_data'][nmr_type] = data

        except Exception as e:
            logger.error(f"Unexpected error processing row in {file_path}: {e}")
            continue

    logger.info(f"Successfully parsed {file_path}")
    return molecules

def parse_SMILES_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)
    except IOError as e:
        logger.error(f"Error opening file {file_path}: {e}")
        return
    except csv.Error as e:
        logger.error(f"CSV error in file {file_path}: {e}")
        return

    for row in data:
        try:
                smiles = row[0].replace(" ","")
                smiles = smiles.strip('\"')

                # Validate SMILES
                if not Chem.MolFromSmiles(smiles):
                    logger.warning(f"Invalid SMILES in {file_path}: {smiles}")
                    continue

        except Exception as e:
            logger.error(f"Unexpected error processing row in {file_path}: {e}")
            continue
