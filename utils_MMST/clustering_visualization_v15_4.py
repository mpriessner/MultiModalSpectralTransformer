# Standard library imports
from io import BytesIO
import base64
import json # For potential future use, good to have

# Third-party imports
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # Using tqdm.auto for better notebook display

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from IPython.display import display, HTML, Image # Image for direct display if needed, HTML for embedding

# --- Configuration for Plotting ---
DEFAULT_FIG_SIZE = (10, 7)
DEFAULT_TITLE_FONTSIZE = 14
DEFAULT_SCATTER_S_MAIN = 100
DEFAULT_SCATTER_S_CONTEXT = 40
DEFAULT_ALPHA_MAIN = 0.7
DEFAULT_ALPHA_CONTEXT = 0.3
DEFAULT_CMAP = 'tab20'

# --- Molecule Processing Helper ---
def _smiles_to_mols_list(smiles_list):
    """Converts a list of SMILES strings to RDKit Mol objects."""
    mols = []
    for smiles in tqdm(smiles_list, desc="Converting SMILES to Mols"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
        mols.append(mol) # Append None if parsing failed, handle downstream
    return mols

def smiles_to_fps(smiles_list, radius=2, nBits=512):
    """Converts a list of SMILES strings to Morgan fingerprints."""
    fps = []
    for smiles in tqdm(smiles_list, desc="Generating Fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(fp)
        else:
            # Append a zero vector or handle as per requirements for failed SMILES
            # For now, appending a zero vector of the same dimension
            fps.append(DataStructs.ExplicitBitVect(nBits)) 
            print(f"Warning: Could not parse SMILES for fingerprint: {smiles}")
    return np.array(fps)

# --- Plotting Core Utilities ---
def _plot_to_html_img(fig):
    """Converts a Matplotlib figure to a base64 encoded HTML image tag."""
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{encoded}">'

def _prepare_plot_data(X_transformed, main_indices_count, group_counts_list, config_nsamples_override=None):
    """
    Prepares data for scatter plot, assigning colors and sizes.
    - main_indices_count: Number of initial points to be plotted distinctly (e.g., in black).
    - group_counts_list: List of integers, where each integer is the size of a subsequent group.
                         Points within each group will share a color.
    - config_nsamples_override: If provided, this overrides config.n_samples for determining main points.
                               This is useful for plot_2D_similarity where main points are targets.
    """
    plot_points = [] # List of dicts: {'x':, 'y':, 'color':, 'size':, 'alpha':, 'label':}
    
    # Determine the number of main (e.g., black) points
    num_main_points = config_nsamples_override if config_nsamples_override is not None else getattr(config, 'n_samples', 0)
    num_main_points = min(num_main_points, len(X_transformed)) # Cap at total points

    # Main points (e.g., source SMILES, targets)
    for i in range(num_main_points):
        plot_points.append({
            'x': X_transformed[i, 0], 'y': X_transformed[i, 1], 
            'color': 'black', 'size': DEFAULT_SCATTER_S_MAIN, 
            'alpha': DEFAULT_ALPHA_MAIN, 'label': f'Source {i+1}' if num_main_points <=10 else None # Avoid too many legend entries
        })

    # Grouped points (e.g., generated SMILES per source)
    current_idx = num_main_points
    num_groups = len(group_counts_list)
    colors = cm.get_cmap(DEFAULT_CMAP, num_groups if num_groups > 0 else 1)

    for group_idx, count in enumerate(group_counts_list):
        group_color = colors(group_idx)
        # The first point of each group (often skipped in original logic, e.g. sublist[1][1:])
        # Here, we assume group_counts_list refers to all points to be colored for that group.
        # If the first point of a group has special meaning (e.g. highest Tanimoto) it should be handled
        # before calling this, or this function needs more complex logic.
        for _ in range(count):
            if current_idx < len(X_transformed):
                plot_points.append({
                    'x': X_transformed[current_idx, 0], 'y': X_transformed[current_idx, 1],
                    'color': group_color, 'size': DEFAULT_SCATTER_S_CONTEXT,
                    'alpha': DEFAULT_ALPHA_CONTEXT, 'label': f'Group {group_idx+1}' if count > 0 and _ == 0 and num_groups <=10 else None
                })
                current_idx += 1
            else:
                break # No more points in X_transformed
        if current_idx >= len(X_transformed):
            break
            
    # Handle any remaining points if group_counts_list doesn't cover all
    # This part might need adjustment based on how group_counts_list is derived
    # For now, assume group_counts_list covers all points after main_indices_count
    
    return plot_points

def _scatter_plot_from_prepared_data(plot_points, title, fig_size=DEFAULT_FIG_SIZE):
    """Generates a scatter plot from prepared plot_points data."""
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Keep track of labels for legend
    handles_labels = {}

    for p in plot_points:
        handle = ax.scatter(p['x'], p['y'], c=[p['color']], s=p['size'], alpha=p['alpha'], label=p.get('label'))
        if p.get('label') and p.get('label') not in handles_labels :
             handles_labels[p.get('label')] = handle # Store handle for unique labels

    ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    if handles_labels: # Only show legend if there are labels
        ax.legend(handles_labels.values(), handles_labels.keys(), loc='best')
    
    return fig


# --- Refactored Plotting Functions ---

def plot_2D_from_data(X_transformed, num_source_points, points_per_group_list, title, fig_size=DEFAULT_FIG_SIZE):
    """
    Generic 2D scatter plot for clustering visualization.
    X_transformed: Data array (N, 2) from PCA, t-SNE, UMAP.
    num_source_points: Number of initial points to be plotted in black (e.g., config.n_samples).
    points_per_group_list: List of integers, where each integer is the number of points in a subsequent colored group.
    title: Plot title.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Cap num_source_points at 30 as in original, or make it a config parameter
    # For now, assuming it's pre-capped or handled by config object
    # num_source_points = min(num_source_points, 30) 
    
    # Plot source points (e.g., original SMILES)
    for i in range(num_source_points):
        if i < X_transformed.shape[0]:
            ax.scatter(X_transformed[i, 0], X_transformed[i, 1], c='black', 
                       s=DEFAULT_SCATTER_S_MAIN, alpha=DEFAULT_ALPHA_MAIN, 
                       label='Source SMILES' if i == 0 else None) # Label only first for legend

    # Plot grouped points (e.g., generated SMILES for each source)
    current_idx = num_source_points
    num_groups = len(points_per_group_list)
    colors = cm.get_cmap(DEFAULT_CMAP, num_groups if num_groups > 0 else 1)

    for group_idx, count in enumerate(points_per_group_list):
        group_color = colors(group_idx)
        # The original code often skipped the first item (e.g., sublist[1][1:])
        # This refactoring assumes points_per_group_list gives the exact number of items to plot for the group
        # If the first item of a group (e.g., highest Tanimoto) needs special handling, 
        # it should be done in the calling function or this logic adapted.
        for i in range(count):
            if current_idx < X_transformed.shape[0]:
                ax.scatter(X_transformed[current_idx, 0], X_transformed[current_idx, 1], 
                           c=[group_color], s=DEFAULT_SCATTER_S_CONTEXT, alpha=DEFAULT_ALPHA_CONTEXT,
                           label=f'Generated Group {group_idx+1}' if i == 0 and group_idx < 5 else None) # Label first of few groups
                current_idx += 1
            else:
                break 
        if current_idx >= X_transformed.shape[0]:
            break
            
    ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only show legend if there are labels
      ax.legend(handles, labels, loc='best', fontsize='small')
    
    return _plot_to_html_img(fig)


def plot_2D_similarity_from_data(X_transformed, num_target_points, num_comparison_points, title, fig_size=DEFAULT_FIG_SIZE):
    """
    2D scatter plot for similarity visualization (targets vs. comparison set).
    X_transformed: Data array (N, 2). First num_target_points are targets.
    num_target_points: Number of target points (plotted in black).
    num_comparison_points: Number of comparison points (plotted in blue, from train set).
    title: Plot title.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot target points
    for i in range(num_target_points):
        if i < X_transformed.shape[0]:
            ax.scatter(X_transformed[i, 0], X_transformed[i, 1], c='black', 
                       s=DEFAULT_SCATTER_S_MAIN, alpha=DEFAULT_ALPHA_MAIN,
                       label='Target SMILES' if i == 0 else None)

    # Plot comparison set points (e.g., from training data)
    # These points start in X_transformed after the target points
    start_comparison_idx = num_target_points
    for j in range(num_comparison_points):
        current_idx = start_comparison_idx + j
        if current_idx < X_transformed.shape[0]:
            ax.scatter(X_transformed[current_idx, 0], X_transformed[current_idx, 1], c='blue', 
                       s=DEFAULT_SCATTER_S_CONTEXT, alpha=DEFAULT_ALPHA_CONTEXT,
                       label='Comparison SMILES' if j == 0 else None)
        else:
            break # No more points

    ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='best', fontsize='small')
        
    return _plot_to_html_img(fig)


def _perform_dimensionality_reduction_and_plot(all_fps, num_source_points, points_per_group_list, config, plot_title_prefix, plot_function):
    """
    Core helper for dimensionality reduction (t-SNE, UMAP, PCA) and plotting.
    plot_function: either plot_2D_from_data or plot_2D_similarity_from_data
    For plot_2D_similarity_from_data, num_source_points is num_target_points, 
    and points_per_group_list is effectively [num_comparison_points].
    """
    # Ensure all_fps is a 2D numpy array
    if not isinstance(all_fps, np.ndarray) or all_fps.ndim != 2:
        # Try to convert if it's a list of lists or similar
        try:
            all_fps = np.array(all_fps, dtype=float)
            if all_fps.ndim != 2: # Check again after conversion
                 raise ValueError("Fingerprint data must be convertible to a 2D numpy array.")
        except ValueError as e:
            raise ValueError(f"Invalid fingerprint data format: {e}")

    if all_fps.shape[0] == 0: # No data to plot
        return "No data provided for plotting.", "No data provided for plotting.", "No data provided for plotting."
    
    # For t-SNE, perplexity should be less than n_samples
    perplexity_value = min(30, all_fps.shape[0] - 1) if all_fps.shape[0] > 1 else 0 # Default RDKit value is 30

    # Dimensionality Reduction
    html_outputs = {}
    reduction_methods = {
        't-SNE': TSNE(n_components=2, random_state=0, perplexity=perplexity_value) if perplexity_value > 0 else None,
        'UMAP': umap.UMAP(n_neighbors=min(15, all_fps.shape[0]-1) if all_fps.shape[0] > 1 else 1, 
                          min_dist=0.1, n_components=2, random_state=0), # n_neighbors must be < n_samples
        'PCA': PCA(n_components=2, random_state=0)
    }

    for name, model in reduction_methods.items():
        if model is None: # Skip if model could not be initialized (e.g. t-SNE with 1 sample)
            html_outputs[name] = f"<p>Could not generate {name} plot (not enough data points).</p>"
            continue
        if all_fps.shape[0] <= model.get_params().get('n_neighbors', 0) and name == 'UMAP': # UMAP specific check
             html_outputs[name] = f"<p>Could not generate {name} plot (n_samples <= n_neighbors for UMAP).</p>"
             continue
        if all_fps.shape[0] <= model.get_params().get('perplexity',0) and name == 't-SNE':
            html_outputs[name] = f"<p>Could not generate {name} plot (n_samples <= perplexity for t-SNE).</p>"
            continue

        X_transformed = model.fit_transform(all_fps)
        
        if plot_function == plot_2D_from_data:
            # num_source_points is typically config.n_samples (number of original SMILES)
            # points_per_group_list is the list of counts of generated SMILES for each original SMILES
            html_outputs[name] = plot_function(X_transformed, num_source_points, points_per_group_list, 
                                               f'{plot_title_prefix} {name} Plot')
        elif plot_function == plot_2D_similarity_from_data:
            # num_source_points is num_target_points
            # points_per_group_list is effectively a single element list: [num_comparison_points]
            # The plot_function itself handles this interpretation.
            num_comparison_points = points_per_group_list[0] if points_per_group_list else 0
            html_outputs[name] = plot_function(X_transformed, num_source_points, num_comparison_points,
                                               f'{plot_title_prefix} {name} Plot')
        else:
            raise ValueError(f"Unsupported plot_function: {plot_function}")


    return html_outputs.get('t-SNE',''), html_outputs.get('UMAP',''), html_outputs.get('PCA','')

# --- Main Clustering and Visualization Functions ---

def _process_results_dict_for_plotting(results_dict, config, skip_first_in_group=True):
    """
    Processes a results dictionary (from MMT/MF) into lists needed for clustering plots.
    Returns:
        all_smiles_list: List of all SMILES (source + generated).
        num_source_smiles: Count of source SMILES.
        points_per_group_list: List of counts of generated SMILES for each source.
    """
    # Clean dictionary keys if necessary (original code removed "0" or 0)
    # This should be handled carefully based on expected key types.
    # For this refactoring, assuming keys are valid and don't need string/int conversion here.
    # results_dict = {k: v for k, v in results_dict.items() if k not in ["0", 0]}


    source_smiles_list = list(results_dict.keys())
    num_source_smiles = len(source_smiles_list)
    
    # config.n_samples often corresponds to num_source_smiles, ensure consistency or clarify
    # Original plot_2D used config.n_samples for black points.
    # Here, num_source_smiles will define the black points.
    # If config.n_samples is different, its role needs to be clarified.
    # For now, we use num_source_smiles as the count for "main" points.
    
    generated_smiles_groups = []
    points_per_group_list = []

    for key in source_smiles_list:
        value = results_dict[key]
        # value structure can be [[item_list_for_group1], [item_list_for_group2]] or [item_list]
        # Original: transformed_list_MMT = [[key, [item[0] for item in value[0]]] for key, value in results_dict.items()]
        # Original: combined_list_MMT = [item for sublist in transformed_list_MMT for item in sublist[1][1:]]
        # Original: number_point_list_MMT = [len(sublist[1]) for sublist in transformed_list_MMT ]
        
        # Assuming value[0] contains the list of generated SMILES strings or [SMILES, prob] pairs
        # And we need the SMILES strings themselves.
        
        current_group_smiles = []
        if isinstance(value, list) and len(value) > 0:
            # Determine if value[0] is a list of SMILES or list of [SMILES, score]
            if isinstance(value[0], list) and len(value[0]) > 0 and isinstance(value[0][0], str): # list of SMILES lists
                 # This case matches plot_cluster_MMT's value[0] being a list of [actual_smiles_str, prob_or_other_metric]
                 # And we need item[0] from that.
                 # Example: value = [ [['smi1',0.9], ['smi2',0.8]], ... ]
                 # Or if value[0] is just a list of SMILES: value = [ ['smi1', 'smi2'], ... ] - less likely based on original
                
                # This logic tries to adapt to plot_cluster_MMT's original processing:
                # `[item[0] for item in value[0]]`
                # And plot_cluster_MF's `value` directly being the list of SMILES.
                
                # Let's assume value format is: { src_smi: [list_of_generated_smiles, ...other_data...] }
                # OR { src_smi: [ [gen_smi1, score1], [gen_smi2, score2] ], ...other_data... }
                # OR { src_smi: direct_list_of_generated_smiles } (for plot_cluster_MF)

                items_to_process = None
                if isinstance(value[0], list) and len(value[0]) > 0:
                    if isinstance(value[0][0], str): # list of SMILES
                        items_to_process = value[0]
                    elif isinstance(value[0][0], (list, tuple)): # list of [SMILES, metric]
                        items_to_process = [item[0] for item in value[0] if isinstance(item, (list, tuple)) and len(item)>0 and isinstance(item[0], str)]
                elif isinstance(value[0], str): # direct list of SMILES (plot_cluster_MF like)
                     items_to_process = value # value itself is the list of generated smiles

                if items_to_process:
                    if skip_first_in_group and len(items_to_process) > 0:
                        current_group_smiles.extend(items_to_process[1:])
                        points_per_group_list.append(len(items_to_process[1:]))
                    else:
                        current_group_smiles.extend(items_to_process)
                        points_per_group_list.append(len(items_to_process))
                    generated_smiles_groups.extend(current_group_smiles)


    all_smiles_list = source_smiles_list + generated_smiles_groups
    return all_smiles_list, num_source_smiles, points_per_group_list


def plot_molecular_clusters(results_dict, config, plot_title_prefix, skip_first_in_group_preprocessing=True):
    """
    Generalized function for MMT and MF style cluster plotting.
    results_dict: Dictionary mapping source SMILES to generated SMILES data.
                  Format: { src_smi: [gen_smi_list_or_detailed_list, ...other_info...] }
                  The exact structure of gen_smi_list_or_detailed_list is handled by _process_results_dict_for_plotting.
    config: Configuration object (must have n_samples if that's used for black points, though num_source_smiles overrides).
    plot_title_prefix: E.g., "MMT Generated Molecules" or "MF Generated Molecules".
    skip_first_in_group_preprocessing: Bool, if True, skips the first generated SMILES in each group (original MMT behavior).
    """
    # Remove "0" or 0 keys if they exist, as in original
    # This is a bit of a code smell from the original, better to ensure consistent key types upstream.
    _results_dict = results_dict.copy() # Avoid modifying original dict
    if "0" in _results_dict: del _results_dict["0"]
    if 0 in _results_dict: del _results_dict[0]

    all_smiles, num_source_smiles, points_per_group = _process_results_dict_for_plotting(
        _results_dict, config, skip_first_in_group=skip_first_in_group_preprocessing
    )

    if not all_smiles:
        print(f"No SMILES data to plot for {plot_title_prefix}.")
        return [], "<p>No SMILES data</p>", "<p>No SMILES data</p>", "<p>No SMILES data</p>"

    all_fps_data = smiles_to_fps(all_smiles)
    
    # num_source_smiles from _process_results_dict_for_plotting is used as n_samples for plot_2D_from_data
    # This means the number of black points will be the number of input source SMILES.
    html_tsne, html_umap, html_pca = _perform_dimensionality_reduction_and_plot(
        all_fps_data, num_source_smiles, points_per_group, config, plot_title_prefix, plot_2D_from_data
    )
    
    # The 'combined_list' returned by original functions was all_smiles excluding sources, 
    # or sometimes all generated smiles including the first ones.
    # Here, all_smiles includes sources. If only generated are needed, slice it: all_smiles[num_source_smiles:]
    generated_smiles_only = all_smiles[num_source_smiles:]

    return generated_smiles_only, html_tsne, html_umap, html_pca


# Wrapper for plot_cluster_MMT
def plot_cluster_MMT(results_dict, config, mode=None, stoi=None, stoi_MF=None, itos=None, itos_MF=None):
    # Unused parameters: mode, stoi, stoi_MF, itos, itos_MF are removed from call to plot_molecular_clusters
    return plot_molecular_clusters(results_dict, config, "MMT Generated Molecules", skip_first_in_group_preprocessing=True)

# Wrapper for plot_cluster_MMT_2 (assuming gen_dict is similar to results_dict)
# The main difference was skip_first_in_group_preprocessing=False
def plot_cluster_MMT_2(config, gen_dict):
     # Original plot_cluster_MMT_2 set config.n_samples = key_len (num_source_smiles)
     # This is implicitly handled as num_source_smiles is passed to plot_2D_from_data via _perform_dimensionality_reduction_and_plot
    return plot_molecular_clusters(gen_dict, config, "MMT Generated Molecules (v2)", skip_first_in_group_preprocessing=False)

# Wrapper for plot_cluster_MF
def plot_cluster_MF(results_dict, config):
    return plot_molecular_clusters(results_dict, config, "MF Generated Molecules", skip_first_in_group_preprocessing=False)


def plot_target_vs_trainset_clusters(smi_list_targets, smi_list_trainset, config):
    """
    Generates and plots clusters for target SMILES vs. a training set sample.
    smi_list_targets: List of target SMILES strings.
    smi_list_trainset: List of training set SMILES for comparison.
    config: Configuration object (must have 'comparision_number' for plot_2D_similarity_from_data,
            and 'data_size' if used by original plot_2D_similarity's colormap, though refactored uses fixed cmap).
    """
    if not smi_list_targets or not smi_list_trainset:
        return [], "<p>Target or trainset SMILES list is empty.</p>", "<p>Empty list</p>", "<p>Empty list</p>"

    num_target_points = len(smi_list_targets)
    # config.comparision_number is used by plot_2D_similarity to determine how many trainset points to show
    # Ensure we don't try to plot more trainset points than available
    num_comparison_points_to_plot = min(getattr(config, 'comparision_number', len(smi_list_trainset)), len(smi_list_trainset))

    # Combined list for fingerprinting and DR. Targets first, then trainset.
    combined_smiles_list = smi_list_targets + smi_list_trainset[:num_comparison_points_to_plot] # Only DR on what will be plotted
    
    all_fps_data = smiles_to_fps(combined_smiles_list)

    # For plot_2D_similarity_from_data, points_per_group_list effectively becomes [num_comparison_points_to_plot]
    # num_source_points becomes num_target_points
    html_tsne, html_umap, html_pca = _perform_dimensionality_reduction_and_plot(
        all_fps_data, 
        num_target_points, 
        [num_comparison_points_to_plot], # This is the "group" of comparison points
        config, 
        "Target vs. Trainset Similarity", 
        plot_2D_similarity_from_data
    )
    
    return combined_smiles_list, html_tsne, html_umap, html_pca


def run_cluster_comparision(config):
    """
    Reads target and training SMILES from CSVs and generates cluster comparison plots.
    config: Must have csv_train_path, csv_SMI_targets, comparision_number.
    """
    try:
        df_train = pd.read_csv(config.csv_train_path)
        df_train = df_train[['SMILES']].sample(n=config.comparision_number, random_state=1) # Ensure 'SMILES' column exists
        smi_list_trainset = list(df_train["SMILES"])
    except Exception as e:
        print(f"Error reading or sampling training data: {e}")
        return None, None, "<p>Error reading train data</p>", "<p>Error reading train data</p>", "<p>Error reading train data</p>"

    try:
        df_target = pd.read_csv(config.csv_SMI_targets)
        df_target = df_target[['SMILES']] # Ensure 'SMILES' column exists
        smi_list_targets = list(df_target["SMILES"])
    except Exception as e:
        print(f"Error reading target data: {e}")
        return df_train, None, "<p>Error reading target data</p>", "<p>Error reading target data</p>", "<p>Error reading target data</p>"

    if not smi_list_targets:
        return df_train, df_target, "<p>No target SMILES found.</p>", "<p>No target SMILES found.</p>", "<p>No target SMILES found.</p>"
    if not smi_list_trainset :
        return df_train, df_target, "<p>No train SMILES selected for comparision.</p>", "<p>No train SMILES selected.</p>", "<p>No train SMILES selected.</p>"
        
    # Note: The original returned combined_list_new from plot_cluster_target, which is combined_smiles_list here.
    # We are not returning it from here, but could if needed. The primary outputs are the HTML plots.
    _, html_TSNE, html_UMAP, html_PCA = plot_target_vs_trainset_clusters(smi_list_targets, smi_list_trainset, config)
    
    return df_train, df_target, html_TSNE, html_UMAP, html_PCA


# --- Molecule Pair Display ---
def _draw_molecule_pair_with_similarity(mol_gen, mol_trg, similarity_score, fig_size=(10,4)):
    """Helper to draw a pair of molecules and their similarity score."""
    fig, axs = plt.subplots(1, 3, figsize=fig_size)
    
    # Generated Molecule
    if mol_gen:
        img_gen = Draw.MolToImage(mol_gen)
        axs[0].imshow(img_gen)
    axs[0].set_title("Generated")
    axs[0].axis('off')

    # Target Molecule
    if mol_trg:
        img_trg = Draw.MolToImage(mol_trg)
        axs[1].imshow(img_trg)
    axs[1].set_title("Target")
    axs[1].axis('off')

    # Similarity Score
    axs[2].text(0.5, 0.5, f'Tanimoto: {similarity_score:.2f}', fontsize=12, ha='center', va='center')
    axs[2].set_title("Similarity")
    axs[2].axis('off')
    
    plt.tight_layout()
    return fig


def display_gen_and_trg_molecules(selected_best_SMI_list, trg_conv_SMI_list, show_number):
    """
    Displays pairs of generated and target molecules with Tanimoto similarity.
    Each pair is shown as a separate plot. For HTML reports, consider aggregating.
    """
    if not selected_best_SMI_list or not trg_conv_SMI_list:
        print("Input SMILES lists are empty.")
        return

    selected_mols = _smiles_to_mols_list(selected_best_SMI_list[:show_number])
    target_mols = _smiles_to_mols_list(trg_conv_SMI_list[:show_number])

    html_outputs = []

    for i in tqdm(range(min(show_number, len(selected_mols), len(target_mols))), desc="Generating molecule pair plots"):
        mol_gen = selected_mols[i]
        mol_trg = target_mols[i]
        
        if mol_gen is None or mol_trg is None:
            html_outputs.append(f"<p>Skipping pair {i+1} due to invalid SMILES.</p>")
            print(f"Skipping pair {i+1} due to invalid SMILES (Gen: {selected_best_SMI_list[i]}, Trg: {trg_conv_SMI_list[i]})")
            continue

        try:
            fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, nBits=512)
            fp_trg = AllChem.GetMorganFingerprintAsBitVect(mol_trg, 2, nBits=512)
            similarity = DataStructs.TanimotoSimilarity(fp_gen, fp_trg)
            
            # Original code used plt.show() in a loop. For reports, better to collect HTML.
            # If plt.show() is desired, this part needs to be conditional.
            fig = _draw_molecule_pair_with_similarity(mol_gen, mol_trg, similarity)
            # html_outputs.append(_plot_to_html_img(fig)) # For embedding in a single HTML page
            plt.show() # Keep original behavior for now if used interactively

        except Exception as e:
            error_message = f"An error occurred for index {i} (Gen: {selected_best_SMI_list[i]}, Trg: {trg_conv_SMI_list[i]}): {e}"
            print(error_message)
            html_outputs.append(f"<p>{error_message}</p>")
            
    # if html_outputs:
    #     display(HTML("".join(html_outputs)))


# --- HTML Colored SMILES and SVG Molecule ---

def generate_colored_html_for_smiles(smiles, probabilities):
    """Generates an HTML string with SMILES characters colored based on probabilities."""
    if len(smiles) != len(probabilities):
        # Fallback or error if lengths don't match
        return f'<span style="font-family: monospace; color: red;">Error: SMILES and probability length mismatch.</span>'

    html_string = '<span style="font-family: monospace;">'
    for char, prob in zip(smiles, probabilities):
        # Color gradient from red (low prob) to green (high prob)
        # Ensuring prob is float for calculation
        try:
            prob_float = float(prob)
            red_value = int((1.0 - prob_float) * 255)
            green_value = int(prob_float * 255)
            color = f'rgb({red_value},{green_value},0)'
        except ValueError: # Fallback color if probability is not a number
            color = 'rgb(128,128,128)' # Grey

        html_string += f'<span style="background-color: {color}">{char}</span>'
    html_string += '</span>'
    return html_string


def _get_atom_color_for_svg(prob):
    """Helper to get RDKit color tuple from probability."""
    try:
        prob_float = float(prob)
        return (1.0 - prob_float, prob_float, 0) # R, G, B
    except ValueError:
        return (0.5, 0.5, 0.5) # Grey for invalid probs


def smiles_to_svg_html(smiles, atom_probabilities=None):
    """
    Creates an SVG image of a molecule, optionally coloring atoms by probabilities.
    Returns it as a base64 encoded HTML image tag.
    atom_probabilities: List of probabilities, one per atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "<p>Invalid SMILES string for SVG generation.</p>"

    d = rdMolDraw2D.MolDraw2DSVG(400, 200) # width, height
    d.drawOptions().useBWAtomPalette() # Black and white palette as a base

    atom_colors = {}
    highlight_atoms_list = []

    if atom_probabilities and mol.GetNumAtoms() == len(atom_probabilities):
        highlight_atoms_list = list(range(mol.GetNumAtoms()))
        for i, prob in enumerate(atom_probabilities):
            atom_colors[i] = _get_atom_color_for_svg(prob)
    
    # DrawMolecule can take highlightAtoms and highlightAtomColors
    d.DrawMolecule(mol, highlightAtoms=highlight_atoms_list, highlightAtomColors=atom_colors if highlight_atoms_list else None)
    d.FinishDrawing()
    
    svg_text = d.GetDrawingText().replace('\n', '')
    b64_svg = base64.b64encode(svg_text.encode('utf-8')).decode()
    return f"<img src='data:image/svg+xml;base64,{b64_svg}'/>"

# Renaming original save_molecule_to_svg to reflect its output and new signature
save_molecule_to_svg_html = smiles_to_svg_html
# Original get_color is similar to _get_atom_color_for_svg, now integrated.
# Original generate_colored_html is renamed to generate_colored_html_for_smiles


# --- Final Check & Placeholder for removed plot_2D and plot_2D_similarity ---
# The original plot_2D and plot_2D_similarity are now replaced by:
# - plot_2D_from_data
# - plot_2D_similarity_from_data
# And the core logic for DR and calling these is in _perform_dimensionality_reduction_and_plot.
# The wrappers (plot_cluster_MMT, etc.) use these new functions.
# This makes the original plot_2D and plot_2D_similarity functions redundant if all calls are updated.
# For now, they are effectively replaced. If direct calls to plot_2D with the old signature
# existed elsewhere and are not updated, that would be an issue.
# The example usage for plot_2D is commented out in the original, suggesting it's not directly called externally.

# Placeholder for any other functions if they were missed or need further attention.
# Ensure all functions from the original file are either refactored, kept, or explicitly noted as removed/replaced.

# Original imports cleaned up at the top.
# tqdm used via tqdm.auto
# Duplicate imports removed.
# IPython.display.Image is kept if direct image display is needed, HTML for embedding.

print("Refactored clustering_visualization_v15_4.py loaded.")
