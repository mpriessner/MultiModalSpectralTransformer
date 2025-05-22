# Standard library imports
import collections
from typing import List, Dict, Tuple, Set, Any, Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdmolfiles, Draw # MolToSmiles, MolFromSmiles also available via Chem
# from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount,EnumerateStereoisomers # Unused
# from rdkit.Chem import SDMolSupplier # Unused
# import os # Unused

# IPython imports (optional, for notebook environments)
# from IPython.display import SVG, display # Can be used if SVG display is desired directly

# --- Molecule Analysis Functions ---

def find_chiral_centers(molecule: Chem.Mol) -> List[int]:
    """Identifies carbon atoms with a specified chiral tag."""
    chiral_centers = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            chiral_centers.append(atom.GetIdx())
    return chiral_centers

def find_carbons_with_relevant_neighbors(molecule: Chem.Mol) -> Dict[int, List[int]]:
    """
    Finds carbon atoms and lists their neighboring carbon atoms 
    if both the atom and its neighbor have attached hydrogens.
    Includes carbons with hydrogens even if they don't couple with any other.
    """
    carbon_connectivity: Dict[int, List[int]] = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 6: # Carbon atom
            atom_idx = atom.GetIdx()
            neighbor_carbons_with_hydrogens = []
            if atom.GetTotalNumHs() > 0: # Source carbon must have hydrogens
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6 and neighbor.GetTotalNumHs() > 0: # Neighbor carbon must also have hydrogens
                        neighbor_carbons_with_hydrogens.append(neighbor.GetIdx())
                carbon_connectivity[atom_idx] = neighbor_carbons_with_hydrogens
    return carbon_connectivity

def find_heavy_atoms_with_hydrogens(molecule: Chem.Mol) -> Dict[int, int]:
    """Creates a dictionary mapping the index of each heavy atom to its count of attached hydrogens."""
    heavy_atom_hydrogens: Dict[int, int] = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:  # Heavy atom
            num_hydrogens = atom.GetTotalNumHs()
            if num_hydrogens > 0:
                heavy_atom_hydrogens[atom.GetIdx()] = num_hydrogens
    return heavy_atom_hydrogens

def map_hydrogen_shifts_to_heavy_atoms(
    all_shifts: List[float], 
    heavy_atom_hydrogens_map: Dict[int, int]
) -> Dict[int, List[float]]:
    """
    Maps a flat list of hydrogen NMR shifts to their respective heavy atoms.
    Assumes `all_shifts` contains heavy atom shifts first, then all hydrogen shifts.
    The order of hydrogen shifts consumed corresponds to heavy atoms sorted by index (descending).
    """
    num_heavy_atoms_with_h = len(heavy_atom_hydrogens_map)
    
    # This calculation of num_total_heavy_atom_shifts seems to imply that all_shifts
    # might contain shifts for heavy atoms not in heavy_atom_hydrogens_map.
    # Or, it's trying to find the starting point of hydrogen shifts.
    # For now, assuming `all_shifts` structure is [C1,C2,..H1,H2,H3..]
    # A clearer approach might be to pass heavy atom shifts and hydrogen shifts separately.
    
    # Original code implies `shifts` is [heavy_atom_shifts_part, hydrogen_shifts_part]
    # Let's assume `all_shifts` is *only* the hydrogen shifts part, or that the heavy atom part was already sliced off.
    # The original `extract_symmetric_hydrogen_shifts` took `shifts` (which was C+H) and did:
    # heavy_atom_shifts = shifts[:num_heavy_atoms] # num_heavy_atoms was len(heavy_atom_dict)
    # hydrogen_shifts_pool = shifts[num_heavy_atoms:]
    # This refactor will assume `all_shifts` is the `hydrogen_shifts_pool` part for clarity.
    # If `all_shifts` is meant to be the combined list, the slicing needs to be done here.

    hydrogen_shifts_pool = list(all_shifts) # Make a mutable copy

    atom_to_h_shifts: Dict[int, List[float]] = {}
    
    # Iterate over heavy atoms, sorted by index descending, to match original consumption order
    for atom_idx, num_hydrogens in sorted(heavy_atom_hydrogens_map.items(), key=lambda x: x[0], reverse=True):
        if num_hydrogens > 0:
            if len(hydrogen_shifts_pool) < num_hydrogens:
                # Not enough shifts left, could raise error or assign empty/placeholder
                # For now, mimics original behavior of potentially assigning fewer shifts
                # if pool is exhausted.
                print(f"Warning: Not enough hydrogen shifts for atom {atom_idx}. Needed {num_hydrogens}, got {len(hydrogen_shifts_pool)}")
                attached_h_shifts = hydrogen_shifts_pool[:] # take all remaining
                hydrogen_shifts_pool = []
            else:
                attached_h_shifts = hydrogen_shifts_pool[-num_hydrogens:]
                hydrogen_shifts_pool = hydrogen_shifts_pool[:-num_hydrogens]
            atom_to_h_shifts[atom_idx] = sorted(attached_h_shifts) # Store shifts, perhaps sorted for consistency
        else:
            atom_to_h_shifts[atom_idx] = []
            
    if hydrogen_shifts_pool: # Any shifts left over?
        print(f"Warning: {len(hydrogen_shifts_pool)} hydrogen shifts remained unassigned.")
        
    # Return sorted by atom index for consistency
    return {idx: atom_to_h_shifts.get(idx, []) for idx in sorted(heavy_atom_hydrogens_map.keys())}


def get_atom_symmetry_groups(smiles: str) -> List[List[int]]:
    """
    Identifies symmetrically equivalent atoms in a molecule.
    Returns a list of groups, where each group is a list of atom indices that are symmetric to each other.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    
    # CanonicalRankAtoms with breakTies=False gives same rank to symmetric atoms
    ranks = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    
    rank_to_indices: Dict[int, List[int]] = collections.defaultdict(list)
    for i, rank in enumerate(ranks):
        rank_to_indices[rank].append(i)
        
    symmetry_groups: List[List[int]] = []
    for rank in rank_to_indices:
        indices = rank_to_indices[rank]
        if len(indices) > 1:
            symmetry_groups.append(sorted(indices))
            
    return symmetry_groups

def average_shifts_for_symmetric_atoms(
    atom_h_shifts_map: Dict[int, List[float]], 
    symmetry_groups: List[List[int]]
) -> Dict[int, List[float]]:
    """
    Averages hydrogen NMR shift values for atoms within the same symmetry group.
    Assumes atom_h_shifts_map contains lists of hydrogen shifts per heavy atom.
    Averages the first hydrogen shift if multiple are present (common for CH2, CH3).
    """
    averaged_atom_h_shifts = atom_h_shifts_map.copy() # Start with a copy

    for group in symmetry_groups:
        if not group:
            continue
        
        shifts_to_average = []
        valid_group_atoms = 0
        for atom_idx in group:
            # Expects list of H shifts, e.g. [2.3, 2.4] for a CH2
            # Typically, for COSY, one representative shift is used or all are considered equivalent
            # This will average the first (or only) shift of each symmetric atom's H list
            h_shifts = atom_h_shifts_map.get(atom_idx)
            if h_shifts: # Atom has hydrogen shifts listed
                shifts_to_average.append(h_shifts[0]) # Average the primary H shift
                valid_group_atoms +=1
        
        if valid_group_atoms > 0:
            avg_shift_val = sum(shifts_to_average) / valid_group_atoms
            for atom_idx in group:
                # Update the list of H shifts for this atom to be [avg_shift_val]
                # This simplifies downstream processing if only one H shift per heavy atom is expected after averaging
                if atom_idx in averaged_atom_h_shifts: # ensure atom was in original map
                     averaged_atom_h_shifts[atom_idx] = [avg_shift_val] 
                # If an atom in a symmetry group didn't have H shifts initially, it won't be added here.
                # This might be desired, or one might want to assign the avg_shift to it.
                # For now, only updates existing entries.

    return averaged_atom_h_shifts

def update_shifts_with_averaged(
    original_shifts: Dict[int, List[float]], 
    averaged_shifts: Dict[int, List[float]]
) -> Dict[int, List[float]]:
    """Updates an original dictionary of shifts with the averaged shifts."""
    updated_shifts = original_shifts.copy()
    updated_shifts.update(averaged_shifts) # Update with averaged values, potentially overwriting
    return updated_shifts

# --- COSY Spectrum Generation and Plotting ---

def _calculate_cosy_cross_peaks(
    heavy_atom_h_shifts: Dict[int, List[float]], 
    carbon_connectivity: Dict[int, List[int]], 
    chiral_centers: List[int]
) -> Set[Tuple[float, float]]:
    """
    Calculates the (H_shift, H_shift) coordinates for COSY cross-peaks.
    heavy_atom_h_shifts: Map from heavy atom index to list of its H shifts.
    carbon_connectivity: Map from C atom index to list of its neighboring C atom indices (for coupling).
    chiral_centers: List of chiral carbon atom indices.
    """
    cross_peaks: Set[Tuple[float, float]] = set()

    for c1_idx, neighbor_c_indices in carbon_connectivity.items():
        h1_shifts = heavy_atom_h_shifts.get(c1_idx, [])
        
        if not h1_shifts: # Carbon c1_idx has no hydrogens or no shifts provided
            continue

        # Add diagonal peaks for all hydrogens on c1_idx
        for h1_s in h1_shifts:
            cross_peaks.add((h1_s, h1_s))
            
        for c2_idx in neighbor_c_indices:
            h2_shifts = heavy_atom_h_shifts.get(c2_idx, [])
            if not h2_shifts: # Carbon c2_idx has no hydrogens or no shifts provided
                continue

            # Determine if coupling involves chiral centers leading to distinct H shifts
            c1_is_chiral = c1_idx in chiral_centers
            c2_is_chiral = c2_idx in chiral_centers # Not used in original logic for peak generation directly but good to have

            # If either carbon is chiral, or if hydrogens on the same carbon are non-equivalent (e.g. CH2 next to chiral)
            # all combinations of H shifts are considered.
            # Original logic: if is_chiral1 or is_chiral2: (is_chiral1 was c1_idx in chiral_centers)
            # This implies that if any of the two carbons in the J-coupling pair is chiral,
            # then all its hydrogens couple with all hydrogens of the other carbon.
            # If neither is chiral, it implies hydrogens on C1 are equivalent, and on C2 are equivalent,
            # so only one H-H coupling is shown (e.g. h1_shifts[0] with h2_shifts[0]).
            
            if c1_is_chiral or c2_is_chiral or len(h1_shifts) > 1 or len(h2_shifts) > 1: 
                # More complex coupling: iterate through all H shifts on C1 and C2
                # This also covers cases like CH2 where H are diastereotopic due to nearby chirality
                for h1_s in h1_shifts:
                    for h2_s in h2_shifts:
                        cross_peaks.add((h1_s, h2_s))
                        cross_peaks.add((h2_s, h1_s)) # Symmetric peak
            else:
                # Simpler coupling (e.g., two CH3 groups or CH groups not affected by chirality)
                # Use only the first (representative) shift from each list
                if h1_shifts and h2_shifts: # Ensure lists are not empty
                    cross_peaks.add((h1_shifts[0], h2_shifts[0]))
                    cross_peaks.add((h2_shifts[0], h1_shifts[0])) # Symmetric peak
                    
    return cross_peaks

def get_cosy_spectrum_data(
    molecule_smiles: str,
    all_nmr_shifts: List[float], # Assumed to be [C_shifts..., H_shifts...]
    num_carbon_shifts: int # Specify how many of the initial shifts are for carbons
) -> Set[Tuple[float, float]]:
    """
    Generates COSY spectrum cross-peak data from molecular structure and NMR shifts.
    
    Args:
        molecule_smiles: SMILES string of the molecule.
        all_nmr_shifts: A flat list containing all NMR shifts, typically carbon shifts followed by all hydrogen shifts.
        num_carbon_shifts: The number of carbon shifts at the beginning of `all_nmr_shifts`.
                           The remaining shifts are assumed to be hydrogen shifts.

    Returns:
        A set of (H_shift, H_shift) tuples representing COSY cross-peaks.
    """
    mol = Chem.MolFromSmiles(molecule_smiles)
    if not mol:
        raise ValueError("Invalid SMILES string provided.")

    # Separate C and H shifts (assuming H shifts follow C shifts in the flat list)
    # This part needs clarification on `all_nmr_shifts` structure.
    # Original `extract_symmetric_hydrogen_shifts` sliced based on `len(heavy_atom_dict)`.
    # If `num_carbon_shifts` is indeed the count of all carbons for which shifts are provided:
    hydrogen_shifts_flat_list = all_nmr_shifts[num_carbon_shifts:]
    
    heavy_atoms_h_counts = find_heavy_atoms_with_hydrogens(mol)
    chiral_centers_indices = find_chiral_centers(mol)
    carbon_connectivity_map = find_carbons_with_relevant_neighbors(mol) # C-C connectivity for J-coupling

    # Map flat H shifts to heavy atoms
    # This is a critical step. `map_hydrogen_shifts_to_heavy_atoms` expects a flat list of H shifts.
    atom_h_shifts_map = map_hydrogen_shifts_to_heavy_atoms(hydrogen_shifts_flat_list, heavy_atoms_h_counts)

    # Consider averaging for symmetric atoms
    symmetry_groups = get_atom_symmetry_groups(molecule_smiles)
    if symmetry_groups:
        atom_h_shifts_map = average_shifts_for_symmetric_atoms(atom_h_shifts_map, symmetry_groups)
        # The original code had `update_shifts_with_averaged` but `average_shifts_for_symmetric_atoms`
        # already returns the updated map.

    cross_peaks = _calculate_cosy_cross_peaks(atom_h_shifts_map, carbon_connectivity_map, chiral_centers_indices)
    return cross_peaks


def plot_cosy_spectrum(
    cross_peaks: Set[Tuple[float, float]], 
    title: str = 'COSY Spectrum',
    xlim: Optional[Tuple[float, float]] = None, 
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """Plots a COSY spectrum from a list of cross-peak coordinates."""
    if not cross_peaks:
        print("No cross-peaks to plot.")
        return

    x_coords = [p[0] for p in cross_peaks]
    y_coords = [p[1] for p in cross_peaks]

    plt.figure(figsize=(8, 7)) # Standard figure size
    plt.scatter(x_coords, y_coords, c='blue', marker='o', label='Cross Peaks', alpha=0.5)
    plt.title(title)
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel('Chemical Shift (ppm)')
    plt.grid(True)
    
    # Standard NMR plot: higher ppm on left/bottom
    if xlim:
        plt.xlim(xlim)
    else: # Auto-scale and invert
        plt.gca().invert_xaxis()
        
    if ylim:
        plt.ylim(ylim)
    else: # Auto-scale and invert
        plt.gca().invert_yaxis()
        
    plt.legend()
    plt.show()


# --- DataFrame Generation ---

def generate_nmr_data_dataframe(
    carbon_shifts: List[float], 
    hydrogen_shifts: List[float]
) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from lists of carbon and hydrogen shifts.
    Assumes a 1:1 correspondence or specific pairing if lists are of different lengths.
    For a typical HSQC-like dataframe, C and H shift lists should correspond.
    The original `generate_COSY_dataframe` took a single `shifts` list of [C,H] pairs.
    This version is more explicit.
    """
    if len(carbon_shifts) != len(hydrogen_shifts):
        # This case needs to be defined: what if C and H lists are different lengths?
        # Pad, truncate, or raise error. For now, assume they should match for HSQC-like data.
        # If it's for COSY (H-H), then only H shifts are needed.
        # The original function name `generate_COSY_dataframe` was misleading if it took C shifts.
        # Renaming to `generate_nmr_data_dataframe` for broader applicability.
        print("Warning: Carbon and Hydrogen shift lists have different lengths. DataFrame might be misaligned.")

    # For an HSQC-like dataframe (F1=Carbon, F2=Hydrogen)
    df = pd.DataFrame({
        'F1_ppm_Carbon': pd.Series(carbon_shifts), # Use pd.Series to handle potential length mismatch if not erroring
        'F2_ppm_Hydrogen': pd.Series(hydrogen_shifts) 
    })
    return df

# Example of how one might call the main functions:
# smiles = "CCO" 
# mol = Chem.MolFromSmiles(smiles)
# num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
# # Dummy shifts: C shifts first, then H shifts in order of attachment (example)
# # This order of H shifts is critical and needs to be consistent with map_hydrogen_shifts_to_heavy_atoms
# example_all_shifts = [58.0, 15.0,  3.5, 3.5, 1.2, 1.2, 1.2] # 2 C shifts, 5 H shifts for ethanol
#
# cosy_peaks = get_cosy_spectrum_data(smiles, example_all_shifts, num_carbons)
# if cosy_peaks:
#    plot_cosy_spectrum(cosy_peaks, title=f"COSY Spectrum for {smiles}")

# Original plot_and_save_cosy_spectrum_with_zoom_no_duplicates was a mix of calculation and plotting.
# It's now separated into get_cosy_spectrum_data (calculation) and plot_cosy_spectrum (plotting).

# Original has_hydrogens function was unused and is removed.
# If needed, it's:
# def has_hydrogens(mol: Chem.Mol, atom_idx: int) -> bool:
#    atom = mol.GetAtomWithIdx(atom_idx)
#    return atom.GetTotalNumHs() > 0

print("Refactored cosy_nmr_reconstruction_v15_4.py loaded.")
