# Standard library imports
import collections
import glob
import os
import time
from typing import List, Dict, Tuple, Any, Optional, Callable

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem, Draw, MolFromSmiles, rdmolfiles, MolToSmiles, SDMolSupplier,
    Descriptors, PandasTools, AddHs, rdDepictor
)
# from rdkit.Chem import EnumerateStereoisomers # Imported but EnumerateStereoisomers itself is used directly
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount, EnumerateStereoisomers
from rdkit.Chem.Draw import rdMolDraw2D

# IPython imports (optional)
from IPython.display import display, SVG

# Local application/library specific imports
# Assuming these paths are valid or will be adjusted based on project structure
from utils_MMT.sgnn_code_pl_v15_4 import main_execute, load_std_mean # Assuming these are the key functions needed
# For dataloaders and clip_functions, direct import might be cleaner if they are also moved to utils_MMST
import utils_MMT.dataloaders_pl_v15_4 as dl 
# import utils_MMT.clip_functions_v15_4 as cl # create_CLIP_dataloaders will be removed from here
import utils_MMT.nmr_calculation_from_dft_v15_4 as ncfd

# Import refactored COSY functions
from utils_MMST.cosy_nmr_reconstruction_v15_4 import (
    get_atom_symmetry_groups, # Replaces find_symmetric_positions
    # The following are needed by run_COSY_generation, which will use the refactored cosy module's versions
    find_chiral_centers,
    find_carbons_with_relevant_neighbors,
    find_heavy_atoms_with_hydrogens,
    map_hydrogen_shifts_to_heavy_atoms, # replaces extract_symmetric_hydrogen_shifts
    # has_hydrogens, # This was unused in cosy_nmr_reconstruction, but might be used here. Let's check.
    average_shifts_for_symmetric_atoms, # replaces average_shifts
    update_shifts_with_averaged,
    get_cosy_spectrum_data, # replaces calculation part of plot_and_save_cosy_spectrum_with_zoom_no_duplicates
    plot_cosy_spectrum # replaces plotting part
)

tqdm.pandas()

# --- SGNN Section ---

def contains_hydrogen_rdkit(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                return True
        return False
    except Exception:
        return False

def calculate_mw(smiles: str) -> Optional[float]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolWt(mol) if mol else None
    except Exception:
        return None

def has_disconnected_smiles(smiles_string: str) -> bool:
    return '.' in smiles_string

def run_sgnn(config: Any) -> pd.DataFrame:
    """Runs SGNN processing on a CSV file of SMILES."""
    graph_representation = "sparsified" # Or from config?
    target_c = "13C"
    train_y_mean_C, train_y_std_C = load_std_mean(target_c, graph_representation)
    target_h = "1H"
    train_y_mean_H, train_y_std_H = load_std_mean(target_h, graph_representation)
    sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)

    batch_size_initial = 128 # Or from config?
    path_csv = config.SGNN_csv_gen_smi
    ml_save_folder = config.SGNN_gen_folder_path

    if not os.path.exists(path_csv):
        print(f"Error: SGNN input CSV not found at {path_csv}")
        return pd.DataFrame()
        
    data_df = pd.read_csv(path_csv)

    # Pre-filtering
    data_df = data_df[data_df['SMILES'].progress_apply(contains_hydrogen_rdkit)]
    data_df = data_df[~data_df['SMILES'].apply(has_disconnected_smiles)]
    data_df['Molecular_Weight'] = data_df['SMILES'].progress_apply(calculate_mw)
    data_df_final = data_df[data_df['Molecular_Weight'] <= config.SGNN_size_filter].copy() # Use .copy() to avoid SettingWithCopyWarning

    if not os.path.exists(ml_save_folder):
        os.makedirs(ml_save_folder)

    # Ensure 'sample-id' column exists, if not, create from index or another column
    if 'sample-id' not in data_df_final.columns:
        if 'zinc_id' in data_df_final.columns: # From original commented code
            data_df_final['sample-id'] = data_df_final['zinc_id']
        else:
            data_df_final['sample-id'] = data_df_final.index.astype(str)


    batch_data_1, failed_ids_1 = main_execute(data_df_final, sgnn_means_stds, ml_save_folder, batch_size_initial)
    print(f"SGNN first pass completed. {len(failed_ids_1)} failed IDs.")

    # Second round for failed molecules with batch_size = 1
    if failed_ids_1:
        data_df_failed = data_df_final[data_df_final['sample-id'].isin(failed_ids_1)].copy()
        if not data_df_failed.empty:
            batch_data_2, failed_ids_2 = main_execute(data_df_failed, sgnn_means_stds, ml_save_folder, 1)
            print(f"SGNN second pass completed for {len(failed_ids_1)} inputs. {len(failed_ids_2)} still failed.")
            combined_df = pd.concat([batch_data_1, batch_data_2], axis=0, ignore_index=True)
        else:
            print("No molecules to retry in the second pass.")
            combined_df = batch_data_1
    else:
        print("No failed IDs in the first pass.")
        combined_df = batch_data_1
        
    return combined_df


# --- 1H NMR Section ---

def read_shifts_from_sdf(file_path: str) -> Dict[int, float]:
    """Reads _Shift property for each atom from an SDF file."""
    supplier = SDMolSupplier(file_path)
    if not supplier or not supplier[0]:
        print(f"Warning: Could not load molecule from {file_path}")
        return {}
    sdf_mol = supplier[0]
    shifts: Dict[int, float] = {}
    for atom in sdf_mol.GetAtoms():
        try:
            atom_shift = atom.GetProp("_Shift")
            shifts[atom.GetIdx()] = float(atom_shift)
        except (KeyError, ValueError):
            # print(f"Warning: Atom {atom.GetIdx()} in {file_path} missing _Shift or invalid format.")
            pass # Atom might not have _Shift property
    return shifts

def lorentzian(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """Calculates Lorentzian peak shape."""
    return (1 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)

def simulate_splitting(
    shifts: np.ndarray, 
    coupling_patterns: List[List[Tuple[float, float]]], 
    gamma: float, 
    spectrometer_frequency: float,
    x_range_padding: float = 1.0,
    num_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates NMR spectrum splitting based on shifts and coupling patterns."""
    if shifts.size == 0:
        return np.array([]), np.array([])
    x_min = shifts.min() - x_range_padding
    x_max = shifts.max() + x_range_padding
    x = np.linspace(x_min, x_max, num_points)
    y = np.zeros_like(x)
    
    for shift_val, pattern in zip(shifts, coupling_patterns):
        peak_signal = np.zeros_like(x)
        for J_coupling, intensity_ratio in pattern:
            # Convert J from Hz to ppm for applying to shift
            j_ppm = J_coupling / spectrometer_frequency 
            peak_signal += intensity_ratio * lorentzian(x, shift_val + j_ppm, gamma)
        y += peak_signal
    return x, y

def get_attached_hydrogens_indices(atom: Chem.Atom) -> List[int]:
    """Returns indices of hydrogen atoms attached to the given atom."""
    return [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H']

def analyze_molecule_for_1h_nmr(mol: Chem.Mol) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], float]]:
    """Analyzes molecule for 1H NMR data, extracting hydrogen info and shifts."""
    hydrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    nmr_data: List[Dict[str, Any]] = []
    assigned_shifts: Dict[Tuple[str, int], float] = {} # Key: (parent_atom_symbol, parent_atom_idx)

    for h_atom in hydrogens:
        parent_atom = h_atom.GetNeighbors()[0] # Assuming H has only one neighbor
        parent_idx = parent_atom.GetIdx()
        parent_symbol = parent_atom.GetSymbol()
        
        group_key = (parent_symbol, parent_idx)
        
        # Store shift if available
        try:
            assigned_shifts[group_key] = float(h_atom.GetProp("_Shift"))
        except (KeyError, ValueError):
            # print(f"Warning: Hydrogen atom {h_atom.GetIdx()} missing _Shift or invalid format.")
            pass

        # Create a label for the hydrogen based on its parent
        # This part might need refinement for complex cases or specific labeling schemes
        attached_h_indices_on_parent = get_attached_hydrogens_indices(parent_atom)
        h_label_suffix = ""
        if len(attached_h_indices_on_parent) > 1:
            try: # Try to find the order of this H among those attached to the same parent
                h_order = attached_h_indices_on_parent.index(h_atom.GetIdx()) + 1
                h_label_suffix = str(h_order)
            except ValueError: # Should not happen if h_atom is correctly identified
                pass

        hydrogen_label = f"{parent_symbol}{parent_idx}H{h_label_suffix}"

        nmr_data.append({
            'atom_idx': h_atom.GetIdx(),
            'parent_atom_idx': parent_idx,
            'parent_atom_symbol': parent_symbol,
            'is_aromatic_parent': parent_atom.GetIsAromatic(),
            'label': hydrogen_label,
            'group_key': group_key # For grouping H shifts by parent atom
        })
            
    return nmr_data, assigned_shifts


def pascals_triangle(n: int) -> List[int]:
    """Generates Pascal's triangle row for binomial coefficients."""
    if n == 0: return [1]
    if n < 0: raise ValueError("Input must be a non-negative integer.")
    row = [1]
    for _ in range(n):
        row = [1] + [row[i] + row[i+1] for i in range(len(row)-1)] + [1]
    return row

def generate_nmr_coupling_pattern(n_equivalent_neighbors: int, J_hz: float) -> List[Tuple[float, float]]:
    """Generates a theoretical coupling pattern (multiplicity) based on n neighbors and J value."""
    if n_equivalent_neighbors < 0: raise ValueError("Number of neighbors cannot be negative.")
    
    multiplicities = pascals_triangle(n_equivalent_neighbors)
    total_intensity = sum(multiplicities) # Should be 2**n_equivalent_neighbors
    
    # Calculate relative J values for the split peaks
    # For a doublet (n=1): -J/2, +J/2. For a triplet (n=2): -J, 0, +J.
    # General form: (k - n/2) * J for k = 0 to n
    j_values = [(k - n_equivalent_neighbors / 2.0) * J_hz for k in range(n_equivalent_neighbors + 1)]
    
    pattern = []
    for i, mult in enumerate(multiplicities):
        pattern.append((j_values[i], mult / total_intensity if total_intensity > 0 else 0))
        
    return pattern


def load_mol_and_assign_shifts_from_sdf(file_path: str) -> Optional[Chem.Mol]:
    """Loads a molecule from an SDF file and assigns NMR shifts stored in a property."""
    try:
        data = PandasTools.LoadSDF(file_path, embedProps=True)
        if data.empty or "ROMol" not in data.columns:
            print(f"Warning: Could not load molecule or ROMol column missing in {file_path}")
            return None
        
        mol = data["ROMol"].iloc[0]
        if not mol: return None
        
        mol = AddHs(mol, addCoords=True) # Add Hs with coordinates initially

        if "averaged_NMR_shifts" in data.columns:
            str_shifts = data["averaged_NMR_shifts"].iloc[0]
            shifts = [float(s) for s in str_shifts.split()]
            
            if len(shifts) == mol.GetNumAtoms():
                for idx, atom in enumerate(mol.GetAtoms()):
                    atom.SetProp("_Shift", str(shifts[idx]))
            else:
                print(f"Warning: Mismatch in number of shifts and atoms in {file_path}.")
        else:
            print(f"Warning: 'averaged_NMR_shifts' not found in SDF {file_path}.")
            # Try to use individual _Shift props if they exist from read_shifts_from_sdf logic
            # This part might be redundant if shifts are already set by another process using _Shift.

        mol = AddHs(mol, addCoords=False) # Ensure Hs are present, but coords might not be needed further
        return mol
    except Exception as e:
        print(f"Error loading molecule and assigning shifts from {file_path}: {e}")
        return None

def average_hydrogen_shifts_on_same_parent(
    nmr_data_list: List[Dict[str, Any]], 
    assigned_shifts_map: Dict[Tuple[str, int], float]
) -> List[Dict[str, Any]]:
    """Averages NMR shifts for hydrogens attached to the same parent atom."""
    
    grouped_raw_shifts: Dict[Tuple[str, int], List[float]] = collections.defaultdict(list)
    for atom_data in nmr_data_list:
        group_key = atom_data['group_key']
        # Use the shift directly from assigned_shifts_map for this group_key
        # This avoids issues if nmr_data_list items don't have 'shift' yet
        if group_key in assigned_shifts_map:
            grouped_raw_shifts[group_key].append(assigned_shifts_map[group_key])

    avg_shifts_by_group: Dict[Tuple[str, int], float] = {}
    for group_key, shifts_list in grouped_raw_shifts.items():
        valid_shifts = [s for s in shifts_list if np.isfinite(s)]
        if valid_shifts:
            avg_shifts_by_group[group_key] = np.mean(valid_shifts)
            
    # Update nmr_data_list with the averaged shift
    updated_nmr_data_list = []
    for atom_data in nmr_data_list:
        group_key = atom_data['group_key']
        if group_key in avg_shifts_by_group:
            # Create a new dict to avoid modifying the original list of dicts in place if it's reused
            new_atom_data = atom_data.copy()
            new_atom_data['avg_shift_on_parent'] = avg_shifts_by_group[group_key]
            updated_nmr_data_list.append(new_atom_data)
        # Else: atom_data for H without a valid shift is dropped here. Decide if that's intended.
        # For now, only H atoms whose parent group had valid shifts are kept.
        
    return updated_nmr_data_list


# TODO: Refactor calculate_couplings_constants by breaking it into smaller rule-based functions
# This function is extremely long and complex. It needs significant refactoring.
# For now, I will keep its structure but add comments and prepare for future refactoring.
def calculate_couplings_constants(
    nmr_data_with_avg_shifts: List[Dict[str, Any]], 
    mol: Chem.Mol
) -> Tuple[List[List[Tuple[float, float]]], List[str], List[float], List[int]]:
    """
    Calculates coupling patterns based on heuristic rules.
    This function is a candidate for major refactoring due to its length and complexity.
    """
    J_aromatic = 8.0  # Hz, typical ortho coupling for aromatics

    coupling_patterns_all: List[List[Tuple[float, float]]] = []
    processed_atom_labels: List[str] = [] # To avoid processing same H group multiple times
    final_shifts_for_pattern: List[float] = []
    num_hydrogens_in_group: List[int] = []

    # Filter out entries without 'avg_shift_on_parent' or non-carbon parents for simplicity in coupling
    # Also, group by 'group_key' to process each unique heavy atom's hydrogens once
    unique_parent_atom_data: Dict[Tuple[str,int], Dict[str,Any]] = {}
    for atom_data_h in nmr_data_with_avg_shifts:
        if 'avg_shift_on_parent' not in atom_data_h:
            continue
        # Only consider hydrogens on carbons for this simplified coupling model
        if not atom_data_h['parent_atom_symbol'].startswith('C'): 
            continue
        
        group_key = atom_data_h['group_key']
        if group_key not in unique_parent_atom_data:
             # Store the first H's data for this parent, assuming all H on same parent are equivalent for coupling purposes here
            unique_parent_atom_data[group_key] = atom_data_h


    for group_key, representative_h_data in unique_parent_atom_data.items():
        parent_atom_rdkit = mol.GetAtomWithIdx(representative_h_data['parent_atom_idx'])
        current_label = representative_h_data['label'].split('H')[0] + "H" # Generic label for the H group (e.g. C5H)
        
        # Skip if this parent atom's H group already processed (e.g. CH2, CH3)
        if current_label in processed_atom_labels:
            continue

        coupling_pattern: Optional[List[Tuple[float, float]]] = None

        # --- Aromatic Hydrogens ---
        if representative_h_data['is_aromatic_parent']:
            # Count hydrogens on *adjacent* aromatic carbons
            num_vicinal_aromatic_h = 0
            for neighbor_atom in parent_atom_rdkit.GetNeighbors():
                if neighbor_atom.GetIsAromatic(): # Should be carbon if parent is aromatic carbon
                    num_vicinal_aromatic_h += neighbor_atom.GetTotalNumHs() # Count H on that neighbor
            
            if num_vicinal_aromatic_h == 0: # E.g. isolated aromatic H or fully substituted neighbors
                coupling_pattern = generate_nmr_coupling_pattern(0, J_aromatic) # Singlet (effectively)
            else:
                # This is a simplification. Real aromatic coupling is more complex (ortho, meta, para).
                # Assuming coupling only to n equivalent ortho hydrogens.
                coupling_pattern = generate_nmr_coupling_pattern(num_vicinal_aromatic_h, J_aromatic)
        
        # --- Aliphatic Hydrogens (Rule-Based) ---
        else:
            # Get carbon neighbors of the parent carbon
            parent_c_neighbors = [n for n in parent_atom_rdkit.GetNeighbors() if n.GetSymbol() == 'C']
            
            # Count hydrogens on these neighboring carbons
            # hydrogen_counts_on_c_neighbors: list of H counts on each neighboring C
            hydrogen_counts_on_c_neighbors = sorted([cn.GetTotalNumHs() for cn in parent_c_neighbors]) # Sorted for consistent rule matching

            # Get bond types to these neighboring carbons
            bond_types_to_c_neighbors = sorted([mol.GetBondBetweenAtoms(parent_atom_rdkit.GetIdx(), cn.GetIdx()).GetBondType() for cn in parent_c_neighbors])


            # Simplified rule matching based on `hydrogen_counts_on_c_neighbors`
            # This section needs to be carefully mapped from the original extensive if-elif block.
            # Example: if hydrogen_counts_on_c_neighbors == [3] and Chem.BondType.SINGLE in bond_types_to_c_neighbors:
            # This implies parent H is coupled to a CH3 group (quartet if parent is CH, etc.)
            # The original code's `hydrogen_counts` was complex. This is a placeholder for that logic.
            
            # Default to singlet if no specific rule matches
            # The number of hydrogens on the *current* parent_atom_rdkit determines the base multiplicity (e.g. CH, CH2, CH3)
            # The coupling is to *neighboring* hydrogens.
            
            # For CH3 group: typically a singlet if no coupled neighbors, or splits based on neighbors.
            # For CH2 group:
            # For CH group:

            # Placeholder: Sum of hydrogens on adjacent carbons for n_neighbors
            # This is a gross simplification of the original detailed rules.
            total_neighboring_hydrogens = sum(hydrogen_counts_on_c_neighbors)
            J_aliphatic_typical = 7.0 # Hz, typical vicinal coupling
            
            # This default pattern is based on N+1 rule with a typical J.
            # Does NOT differentiate CH, CH2, CH3, or complex second-order effects.
            coupling_pattern = generate_nmr_coupling_pattern(total_neighboring_hydrogens, J_aliphatic_typical)


        if coupling_pattern is not None:
            coupling_patterns_all.append(coupling_pattern)
            processed_atom_labels.append(current_label)
            final_shifts_for_pattern.append(representative_h_data['avg_shift_on_parent'])
            num_hydrogens_in_group.append(parent_atom_rdkit.GetTotalNumHs()) # Actual H count on this parent
        # else:
            # print(f"No coupling pattern determined for {current_label} with H counts {hydrogen_counts_on_c_neighbors}")

    return coupling_patterns_all, processed_atom_labels, final_shifts_for_pattern, num_hydrogens_in_group


def create_nmr_plot_matplotlib(
    shifts_ppm: List[float], 
    coupling_patterns: List[List[Tuple[float, float]]], 
    atom_labels: List[str],
    gamma: float, 
    spectrometer_frequency_mhz: float,
    title: str = "Simulated 1H NMR Spectrum"
) -> None:
    """Plots 1H NMR spectrum using Matplotlib."""
    if not shifts_ppm:
        print("No shifts provided for plotting.")
        return

    # Simulate the full spectrum by summing individual peak patterns
    # Ensure shifts_ppm is numpy array for vectorized operations in simulate_splitting
    x, y = simulate_splitting(np.array(shifts_ppm), coupling_patterns, gamma, spectrometer_frequency_mhz)

    if x.size == 0:
        print("Spectrum simulation resulted in no data points.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.xlabel('Chemical shift (ppm)')
    plt.ylabel('Intensity')
    plt.title(title)
    
    # Add labels at peak maxima (approximate)
    # This needs a more robust way to find actual peak maxima in the convolved spectrum `y`
    for shift_val, label_text in zip(shifts_ppm, atom_labels):
        # Find closest x point to shift_val for y_max estimation
        idx = (np.abs(x - shift_val)).argmin()
        y_max_at_shift = y[idx] # Approximate intensity at the original shift position
        plt.text(shift_val, y_max_at_shift, label_text, ha='center', va='bottom', fontsize=8, rotation=45)

    plt.gca().invert_xaxis() # Standard NMR display
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


# create_plot_NMR_interactiv using Plotly is omitted for now as Plotly is not a standard dependency
# and the function relied on global variables. If needed, it should be refactored similarly.

def create_labeled_rdkit_svg(mol: Chem.Mol, atom_labels_map: Dict[int, str]) -> Optional[str]:
    """Creates an SVG image of the molecule with specified atom labels."""
    try:
        # Ensure 2D coordinates
        if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D():
             rdDepictor.Compute2DCoords(mol)
        
        drawer = rdMolDraw2D.MolDraw2DSVG(600, 300) # width, height
        opts = drawer.drawOptions()

        for atom_idx, label in atom_labels_map.items():
            opts.atomLabels[atom_idx] = label
        
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        print(f"Error generating RDKit SVG: {e}")
        return None

def prepare_peak_data_for_output(
    shifts_ppm: List[float], 
    coupling_patterns: List[List[Tuple[float, float]]], 
    atom_labels: List[str], # Labels for the original (center) shifts
    spectrometer_frequency_mhz: float
) -> List[Tuple[float, float]]:
    """Generates a list of (shift, intensity) tuples for all individual peaks after splitting."""
    all_peaks_data: List[Tuple[float, float]] = []
    
    for center_shift, pattern, _ in zip(shifts_ppm, coupling_patterns, atom_labels):
        if center_shift == 0.0: # Skip padding values if used
            continue
        for j_offset_hz, intensity_ratio in pattern:
            actual_shift = center_shift + (j_offset_hz / spectrometer_frequency_mhz)
            all_peaks_data.append((actual_shift, intensity_ratio))
            
    return sorted(list(set(all_peaks_data)), key=lambda p: p[0]) # Unique, sorted by shift


def run_1h_nmr_generation(config: Any) -> Tuple[pd.DataFrame, str]:
    """Main function to generate 1H NMR data from SGNN outputs."""
    folder_path = config.SGNN_gen_folder_path
    if not os.path.isdir(folder_path):
        print(f"Error: SGNN generation folder not found at {folder_path}")
        return pd.DataFrame(), ""
        
    nmr_files = glob.glob(os.path.join(folder_path, "NMR_*")) # More specific glob
    nmr_files = [f for f in nmr_files if not f.endswith(".mol")] # Exclude .mol files if any
    nmr_files = sorted(nmr_files)

    # Settings from original code
    spectrometer_frequency_mhz = 400.0
    gamma_hz = 0.01 # Linewidth for Lorentzian peaks (in ppm for lorentzian function)
    # Convert gamma to ppm for lorentzian function: gamma_ppm = gamma_hz / spectrometer_frequency_mhz
    gamma_ppm = gamma_hz / spectrometer_frequency_mhz


    results_smiles: List[str] = []
    results_peak_data: List[List[Tuple[float, float]]] = [] # List of (shift, intensity) tuples
    results_sample_ids: List[str] = []

    for file_path in tqdm(nmr_files, desc="Processing 1H NMR"):
        try:
            mol_with_shifts = load_mol_and_assign_shifts_from_sdf(file_path)
            if not mol_with_shifts:
                print(f"Skipping {file_path} due to loading error.")
                continue

            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            nmr_data_raw, assigned_shifts_map = analyze_molecule_for_1h_nmr(mol_with_shifts)
            nmr_data_with_avg_parent_shifts = average_hydrogen_shifts_on_same_parent(nmr_data_raw, assigned_shifts_map)
            
            # This is the complex part.
            coupling_patterns, processed_atom_labels, final_center_shifts, _ = \
                calculate_couplings_constants(nmr_data_with_avg_parent_shifts, mol_with_shifts)

            if config.get("plot_1H_NMR", False): # Make plotting configurable
                 create_nmr_plot_matplotlib(final_center_shifts, coupling_patterns, processed_atom_labels, 
                                           gamma_ppm, spectrometer_frequency_mhz, 
                                           title=f"1H NMR for {sample_id}")
            
            # if config.get("show_1H_labeled_structure", False):
            #     # Create labels for SVG: map atom index to string label
            #     atom_idx_to_label_map = {data['parent_atom_idx']: data['label'].split('H')[0] + "H" 
            #                              for data in nmr_data_with_avg_parent_shifts}
            #     svg_str = create_labeled_rdkit_svg(mol_with_shifts, atom_idx_to_label_map)
            #     if svg_str: display(SVG(svg_str))

            peak_tuples_for_sample = prepare_peak_data_for_output(
                final_center_shifts, coupling_patterns, processed_atom_labels, spectrometer_frequency_mhz
            )
            
            if peak_tuples_for_sample:
                mol_no_hs = Chem.RemoveHs(mol_with_shifts)
                smi = Chem.MolToSmiles(mol_no_hs)
                results_smiles.append(smi)
                results_peak_data.append(peak_tuples_for_sample)
                results_sample_ids.append(sample_id)
            else:
                print(f"No peak data generated for {sample_id} ({file_path})")

        except Exception as e:
            print(f"Error processing {file_path} for 1H NMR: {e}")
            import traceback
            traceback.print_exc()


    df_1h = pd.DataFrame({
        'SMILES': results_smiles,
        'shifts_intensities': results_peak_data, # Storing as list of [shift, intensity] tuples
        'sample-id': results_sample_ids,
    })
    df_1h.reset_index(drop=True, inplace=True)
    
    csv_1h_path = os.path.join(config.SGNN_csv_save_folder, f"data_1H_{config.ran_num}.csv")
    if not os.path.exists(config.SGNN_csv_save_folder):
        os.makedirs(config.SGNN_csv_save_folder)
    df_1h.to_csv(csv_1h_path, index=False)
    
    return df_1h, csv_1h_path


# --- 13C NMR Section ---

def consolidate_symmetric_carbon_shifts(
    initial_shifts: List[float], 
    symmetry_groups: List[List[int]]
) -> List[float]:
    """Averages shifts for symmetric carbon atoms based on their indices."""
    # This function replaces the original `consolidate_peaks`
    # Assumes `initial_shifts` is a list where index corresponds to atom index for carbons.
    
    if not symmetry_groups:
        return initial_shifts

    # Create a mutable copy if initial_shifts should not be changed, or if it's a numpy array.
    # If it's a list, direct modification is fine if intended.
    averaged_shifts_list = list(initial_shifts) # Work on a copy

    for group_indices in symmetry_groups:
        if not group_indices: continue
        
        group_shifts = [averaged_shifts_list[i] for i in group_indices if i < len(averaged_shifts_list)]
        if not group_shifts: continue # Should not happen if indices are correct

        avg_value = sum(group_shifts) / len(group_shifts)
        
        for i in group_indices:
            if i < len(averaged_shifts_list):
                averaged_shifts_list[i] = avg_value
    return averaged_shifts_list


def run_13c_nmr_generation(config: Any) -> Tuple[pd.DataFrame, str]:
    """Main function to generate 13C NMR data."""
    folder_path = config.SGNN_gen_folder_path
    if not os.path.isdir(folder_path):
        print(f"Error: SGNN generation folder not found at {folder_path}")
        return pd.DataFrame(), ""

    nmr_files = glob.glob(os.path.join(folder_path, "NMR_*"))
    nmr_files = [f for f in nmr_files if not f.endswith(".mol")]
    nmr_files = sorted(nmr_files)

    results_smiles: List[str] = []
    results_peak_data: List[List[float]] = [] # List of unique, sorted C shifts
    results_sample_ids: List[str] = []

    for file_path in tqdm(nmr_files, desc="Processing 13C NMR"):
        try:
            mol_supplier = SDMolSupplier(file_path)
            if not mol_supplier or not mol_supplier[0]:
                print(f"Skipping {file_path}, cannot load molecule.")
                continue
            mol = mol_supplier[0]

            # Generate a canonical SMILES with stereochemistry for symmetry perception
            # Consider AddHs for consistent stereoisomer enumeration if input has H issues
            # mol_with_hs_for_stereo = AddHs(mol) 
            # isomers = tuple(EnumerateStereoisomers(mol_with_hs_for_stereo)) 
            isomers = tuple(EnumerateStereoisomers.EnumerateStereoisomers(mol)) # Corrected call
            stereo_smi = Chem.MolToSmiles(isomers[0], isomericSmiles=True) if isomers else Chem.MolToSmiles(mol, isomericSmiles=True)


            if 'averaged_NMR_shifts' not in mol.GetPropsAsDict():
                 print(f"Skipping {file_path}, 'averaged_NMR_shifts' not found.")
                 continue
            averaged_nmr_shifts_str = mol.GetProp('averaged_NMR_shifts')
            all_atom_shifts = list(map(float, averaged_nmr_shifts_str.split()))

            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Extract shifts for heavy atoms only (carbons for 13C NMR)
            heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
            # This assumes shifts in averaged_NMR_shifts are ordered same as GetAtoms(), including Hs
            # Need to map shifts to heavy atoms correctly.
            # A robust way: use the atom._Shift property if set by SGNN, or map based on atom order in SDF.
            # For now, assume the original logic's slicing was correct for non-hydrogen atoms:
            
            num_heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != 'H')
            heavy_atom_shifts_from_list = all_atom_shifts[:num_heavy_atoms] # This slice is risky if H shifts are interspersed

            # A safer way if shifts are ordered C, H:
            # Get C atom indices and their corresponding shifts
            carbon_atom_shifts_map: Dict[int, float] = {}
            current_shift_idx = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6: # Carbon
                    if current_shift_idx < len(heavy_atom_shifts_from_list):
                         carbon_atom_shifts_map[atom.GetIdx()] = heavy_atom_shifts_from_list[current_shift_idx]
                         current_shift_idx += 1
                    else: # Should not happen if heavy_atom_shifts_from_list is correct
                         print(f"Warning: Not enough shifts for Carbon atom {atom.GetIdx()} in {sample_id}")
            
            # If using carbon_atom_shifts_map, consolidate_symmetric_carbon_shifts needs adaptation
            # For now, let's stick to list if possible, assuming it's correctly ordered for carbons
            # The original `consolidate_peaks` took a list `sample_shifts` which was `all_atom_shifts`
            # and `sym_dupl_lists` which were indices.

            # Get symmetry groups for the molecule (indices are based on the `mol` object)
            symmetry_groups_indices = get_atom_symmetry_groups(stereo_smi) 
            
            # Filter symmetry groups to only include Carbon atoms
            carbon_symmetry_groups = []
            for group in symmetry_groups_indices:
                carbon_group = [idx for idx in group if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
                if len(carbon_group) > 1: # Only keep if multiple carbons are symmetric
                    carbon_symmetry_groups.append(carbon_group)

            # Consolidate shifts based on symmetry
            # `heavy_atom_shifts_from_list` is assumed to be ordered list of shifts for heavy atoms
            # Indices in `carbon_symmetry_groups` must map correctly to this list.
            # This is tricky. Original `consolidate_peaks` worked on `all_atom_shifts` list.
            # Let's use a dictionary approach for robustness:
            
            # Re-map all_atom_shifts to a dictionary: {atom_idx: shift} for Carbons
            atom_idx_to_shift_map = {
                atom.GetIdx(): all_atom_shifts[atom.GetIdx()] 
                for atom in mol.GetAtoms() 
                if atom.GetAtomicNum() == 6 and atom.GetIdx() < len(all_atom_shifts)
            }

            # Average shifts for symmetric carbons
            temp_averaged_shifts = atom_idx_to_shift_map.copy()
            for group in carbon_symmetry_groups:
                group_c_shifts = [temp_averaged_shifts[idx] for idx in group if idx in temp_averaged_shifts]
                if group_c_shifts:
                    avg_c_shift = sum(group_c_shifts) / len(group_c_shifts)
                    for idx in group:
                        if idx in temp_averaged_shifts: temp_averaged_shifts[idx] = avg_c_shift
            
            # Get final list of carbon shifts, unique and sorted
            final_carbon_shifts = sorted(list(set(s for s in temp_averaged_shifts.values() if s != 0.0)))


            mol_no_hs = Chem.RemoveHs(mol)
            smi_out = Chem.MolToSmiles(mol_no_hs)     

            results_smiles.append(smi_out)
            results_peak_data.append(final_carbon_shifts)
            results_sample_ids.append(sample_id)

        except Exception as e:
            print(f"Error processing {file_path} for 13C NMR: {e}")
            import traceback
            traceback.print_exc()

    df_13c = pd.DataFrame({
        'SMILES': results_smiles,
        'shifts': results_peak_data, # List of unique sorted C shifts
        'sample-id': results_sample_ids,
    })
    df_13c.reset_index(drop=True, inplace=True)

    csv_13c_path = os.path.join(config.SGNN_csv_save_folder, f"data_13C_{config.ran_num}.csv")
    if not os.path.exists(config.SGNN_csv_save_folder):
        os.makedirs(config.SGNN_csv_save_folder)
    df_13c.to_csv(csv_13c_path, index=False)
    
    return df_13c, csv_13c_path


# --- COSY Section ---
# The COSY helper functions (find_chiral_centers, etc.) were duplicated.
# They are now imported from the refactored cosy_nmr_reconstruction_v15_4.py.

def run_cosy_generation(config: Any) -> Tuple[pd.DataFrame, str]:
    """Main function to generate COSY NMR data."""
    folder_path = config.SGNN_gen_folder_path
    if not os.path.isdir(folder_path):
        print(f"Error: SGNN generation folder not found at {folder_path}")
        return pd.DataFrame(), ""

    nmr_files = glob.glob(os.path.join(folder_path, "NMR_*"))
    nmr_files = [f for f in nmr_files if not f.endswith(".mol")]
    nmr_files = sorted(nmr_files)

    results_smiles: List[str] = []
    results_peak_data: List[List[Tuple[float, float]]] = [] # List of (H_shift, H_shift) tuples
    results_sample_ids: List[str] = []

    for file_path in tqdm(nmr_files, desc="Processing COSY"):
        try:
            mol_supplier = SDMolSupplier(file_path)
            if not mol_supplier or not mol_supplier[0]:
                print(f"Skipping {file_path}, cannot load molecule.")
                continue
            mol = mol_supplier[0]
            
            # Needed for get_cosy_spectrum_data if it expects SMILES
            # Also for symmetry perception
            # mol_for_smiles = Chem.RemoveHs(mol) # Create SMILES from mol without Hs for canonical representation
            isomers = tuple(EnumerateStereoisomers.EnumerateStereoisomers(mol))
            stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True) if isomers else Chem.MolToSmiles(mol,isomericSmiles=True)


            if 'averaged_NMR_shifts' not in mol.GetPropsAsDict():
                 print(f"Skipping {file_path}, 'averaged_NMR_shifts' not found.")
                 continue
            averaged_nmr_shifts_str = mol.GetProp('averaged_NMR_shifts')
            all_atom_shifts_list = list(map(float, averaged_nmr_shifts_str.split()))
            
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Determine num_carbon_shifts for get_cosy_spectrum_data
            # This assumes shifts are C shifts first, then H shifts
            num_carbon_atoms_in_mol = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            # num_heavy_atoms_in_mol = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
            # The `get_cosy_spectrum_data` needs to know how many of the leading shifts in `all_atom_shifts_list`
            # are for carbons, so it can correctly find the start of hydrogen shifts.
            # This is a bit fragile. A better SGNN output would separate C and H shifts or provide clearer mapping.

            cosy_cross_peaks_set = get_cosy_spectrum_data(
                stereo_smi, # Pass SMILES for internal Mol conversion and symmetry
                all_atom_shifts_list,
                num_carbon_atoms_in_mol # Assuming leading shifts are for these carbons
            )
            
            # Optional: Plotting (using the imported plot_cosy_spectrum)
            # if config.get("plot_COSY", False) and cosy_cross_peaks_set:
            #    plot_cosy_spectrum(cosy_cross_peaks_set, title=f"COSY for {sample_id}")

            cosy_shifts_list = sorted(list(cosy_cross_peaks_set), key=lambda x: (x[0], x[1]))

            mol_no_hs = Chem.RemoveHs(mol)
            smi_out = Chem.MolToSmiles(mol_no_hs)     

            results_smiles.append(smi_out)
            results_peak_data.append(cosy_shifts_list)
            results_sample_ids.append(sample_id)

        except Exception as e:
            print(f"Error processing {file_path} for COSY: {e}")
            import traceback
            traceback.print_exc()
            
    df_cosy = pd.DataFrame({
        'SMILES': results_smiles,
        'shifts': results_peak_data, # List of [H_shift1, H_shift2] tuples
        'sample-id': results_sample_ids,
    })
    df_cosy.reset_index(drop=True, inplace=True)
    
    csv_cosy_path = os.path.join(config.SGNN_csv_save_folder, f"data_COSY_{config.ran_num}.csv")
    if not os.path.exists(config.SGNN_csv_save_folder):
        os.makedirs(config.SGNN_csv_save_folder)
    df_cosy.to_csv(csv_cosy_path, index=False)
    
    return df_cosy, csv_cosy_path


# --- HSQC Section ---

def run_hsqc_generation(config: Any) -> Tuple[pd.DataFrame, str]:
    """Main function to generate HSQC NMR data."""
    folder_path = config.SGNN_gen_folder_path
    if not os.path.isdir(folder_path):
        print(f"Error: SGNN generation folder not found at {folder_path}")
        return pd.DataFrame(), ""
        
    nmr_files = glob.glob(os.path.join(folder_path, "NMR_*"))
    nmr_files = [f for f in nmr_files if not f.endswith(".mol")]
    nmr_files = sorted(nmr_files)

    results_smiles: List[str] = []
    results_peak_data: List[List[Tuple[float, float]]]] = [] # List of [H_shift (F2), C_shift (F1)]
    results_sample_ids: List[str] = []

    for file_path in tqdm(nmr_files, desc="Processing HSQC"):
        try:
            # ncfd.load_dft_dft_comparison is expected to return a DataFrame with 'F1 (ppm)' and 'F2 (ppm)'
            sample_df = ncfd.load_dft_dft_comparison(file_path)
            if sample_df.empty:
                print(f"No data loaded from {file_path} by load_dft_dft_comparison.")
                continue

            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Ensure required columns exist
            if 'F1 (ppm)' not in sample_df.columns or 'F2 (ppm)' not in sample_df.columns:
                print(f"Missing F1/F2 columns in data from {file_path}")
                continue

            hsqc_shifts_list = sample_df.apply(lambda row: (row['F2 (ppm)'], row['F1 (ppm)']), axis=1).tolist()
            # Sort by H shift (F2), then C shift (F1)
            hsqc_shifts_list = sorted(hsqc_shifts_list, key=lambda x: (x[0], x[1])) 

            # Get SMILES for the output DataFrame
            mol_supplier = SDMolSupplier(file_path) # Reloading mol to get SMILES
            if mol_supplier and mol_supplier[0]:
                mol = mol_supplier[0]   
                mol = Chem.RemoveHs(mol)
                smi_out = Chem.MolToSmiles(mol)   
            else:
                smi_out = "N/A" # Fallback SMILES
                print(f"Could not load molecule from {file_path} to generate SMILES for HSQC output.")


            results_smiles.append(smi_out)
            results_peak_data.append(hsqc_shifts_list)
            results_sample_ids.append(sample_id)
        except Exception as e:
            print(f"Error processing {file_path} for HSQC: {e}")
            import traceback
            traceback.print_exc()

    df_hsqc = pd.DataFrame({
        'SMILES': results_smiles,
        'shifts': results_peak_data, # List of [H_shift, C_shift] tuples
        'sample-id': results_sample_ids,
    })
    df_hsqc.reset_index(drop=True, inplace=True)
    
    csv_hsqc_path = os.path.join(config.SGNN_csv_save_folder, f"data_HSQC_{config.ran_num}.csv")
    if not os.path.exists(config.SGNN_csv_save_folder):
        os.makedirs(config.SGNN_csv_save_folder)
    df_hsqc.to_csv(csv_hsqc_path, index=False) # Original did not have index=False, adding for consistency
    
    return df_hsqc, csv_hsqc_path


# --- Main Orchestration Function ---

def main_run_data_generation(config: Any) -> Tuple[Optional[pd.DataFrame], ...]:
    """Runs all data generation pipelines."""
    # Run SGNN
    combined_df_sgnn = run_sgnn(config)
    print("\033[1m\033[33mrun_sgnn: DONE\033[0m")

    # Run 1H NMR Generation
    data_1h, csv_1h_path = run_1h_nmr_generation(config)
    print("\033[1m\033[33mrun_1H_generation: DONE\033[0m")

    # Run 13C NMR Generation
    data_13c, csv_13c_path = run_13c_nmr_generation(config)
    print("\033[1m\033[33mrun_13C_generation: DONE\033[0m")
    
    # Run COSY Generation
    data_cosy, csv_cosy_path = run_cosy_generation(config)
    print("\033[1m\033[33mrun_COSY_generation: DONE\033[0m")

    # Run HSQC Generation
    data_hsqc, csv_hsqc_path = run_hsqc_generation(config)
    print("\033[1m\033[33mrun_HSQC_generation: DONE\033[0m")
    
    return (combined_df_sgnn, 
            data_1h, data_13c, data_cosy, data_hsqc, 
            csv_1h_path, csv_13c_path, csv_cosy_path, csv_hsqc_path)


# Removed create_CLIP_dataloaders as it's redundant with clip_functions_v15_4.py

print("Refactored data_generation_v15_4.py loaded.")
