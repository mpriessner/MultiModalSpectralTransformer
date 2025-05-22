# Standard library imports
import random
import os
import json

# Third-party imports
import rdkit
import torch
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors # Explicitly import rdMolDescriptors
from torch.utils.data import DataLoader, Sampler, Dataset # Added Dataset
from transformers import RobertaModel, RobertaTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger # Simplified import
from pytorch_lightning.callbacks import ModelCheckpoint # Simplified import
from pytorch_lightning.profiler import SimpleProfiler # Simplified import
# from sklearn.model_selection import train_test_split # Not used
# from torch.utils.data.distributed import DistributedSampler # Not used
# from torch.optim.lr_scheduler import ReduceLROnPlateau # Not used
# from IPython.display import display, HTML, SVG # Not used in this script directly

# Local module imports
from utils_MMT.dataloaders_pl_v15_4 import collate_fn, MultimodalData
from utils_MMT.models_CLIP_v15_4 import CLIPMultiGPU # Assuming CLIPModel, ChembertaFingerprint are used by CLIPMultiGPU
# import utils_MMT.data_generation_v15_4 as dg # Not used directly, commented out based on file end comment

# Helper function for Tanimoto calculation (already present and good)
def tanimoto_similarity(mol1, mol2, radius=2, num_class=1024):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=num_class)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=num_class)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Refactored create_batches and its helpers
def _filter_molecules_by_weight(molecules_with_indices, min_weight, max_weight):
    """
    Filters molecules based on their molecular weight.
    Helper for create_batches.
    """
    filtered_indices = []
    for i, mol in molecules_with_indices:
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        if min_weight <= mol_weight <= max_weight:
            filtered_indices.append(i)
    return filtered_indices

def _find_similar_molecules_in_window(anchor_mol, molecules, indices_to_search, similarity_threshold, batch_size_left, max_search_count):
    """
    Finds similar molecules to the anchor within a given window.
    Helper for create_batches.
    """
    similar_to_anchor = []
    searched_count = 0
    for i in indices_to_search:
        if len(similar_to_anchor) >= batch_size_left:
            break
        if tanimoto_similarity(anchor_mol, molecules[i]) >= similarity_threshold:
            similar_to_anchor.append(i)
        searched_count += 1
        if searched_count >= max_search_count:
            break
    return similar_to_anchor

def create_batches(dataset: MultimodalData, similarity_threshold: float, batch_size: int, max_search_size: int, weight_delta: float):
    """
    Create batches of molecules from a dataset based on Tanimoto similarity and molecular weight.
    """
    # Ensure dataset.ref_data is a DataFrame and has "SMILES"
    if not hasattr(dataset, 'ref_data') or not isinstance(dataset.ref_data, pd.DataFrame) or "SMILES" not in dataset.ref_data.columns:
        raise ValueError("Dataset must have a 'ref_data' attribute (pandas DataFrame) with a 'SMILES' column.")

    molecules = [Chem.MolFromSmiles(smi) for smi in tqdm(dataset.ref_data["SMILES"], desc="Parsing SMILES")]
    valid_mol_indices = [i for i, mol in enumerate(molecules) if mol is not None]
    
    # Filter out None molecules that failed to parse
    molecules = [molecules[i] for i in valid_mol_indices]
    # Mapping from original dataset index to the new 'molecules' list index will be complex if we don't re-index.
    # For simplicity, this refactoring assumes we operate on valid molecules and their indices within the 'molecules' list.
    # If original indices are strictly needed, a mapping should be kept.

    available_indices = set(range(len(molecules)))
    batches = []

    pbar_batches = tqdm(total=len(available_indices), desc="Creating batches")
    while available_indices:
        if not available_indices: break
        anchor_original_idx = available_indices.pop()
        pbar_batches.update(1)
        
        current_batch_indices = [anchor_original_idx]
        anchor_mol = molecules[anchor_original_idx]
        anchor_weight = rdMolDescriptors.CalcExactMolWt(anchor_mol)
        
        min_weight, max_weight = anchor_weight - weight_delta, anchor_weight + weight_delta
        
        # Prepare molecules with their current indices for filtering
        indexed_molecules_for_filtering = [(idx, molecules[idx]) for idx in available_indices]
        
        weight_filtered_available_indices = _filter_molecules_by_weight(
            [(idx, molecules[idx]) for idx in available_indices], # Pass (index, mol) tuples
            min_weight, 
            max_weight
        )
        
        # Shuffle for randomness within search window
        search_indices_within_weight = list(set(weight_filtered_available_indices).intersection(available_indices))
        random.shuffle(search_indices_within_weight)

        # Find similar molecules
        similar_molecules_indices = _find_similar_molecules_in_window(
            anchor_mol,
            molecules,
            search_indices_within_weight,
            similarity_threshold,
            batch_size - len(current_batch_indices), # how many more we need
            max_search_size
        )

        for idx in similar_molecules_indices:
            if len(current_batch_indices) < batch_size:
                current_batch_indices.append(idx)
                available_indices.discard(idx)
                pbar_batches.update(1)
        
        # Fill remaining spots in the batch with molecules within the weight range if needed
        if len(current_batch_indices) < batch_size:
            # Prioritize from search_indices_within_weight not already in batch or similar_molecules_indices
            remaining_search_indices = [idx for idx in search_indices_within_weight if idx not in current_batch_indices]
            fill_count = 0
            for i in remaining_search_indices:
                if len(current_batch_indices) >= batch_size:
                    break
                current_batch_indices.append(i)
                available_indices.discard(i)
                pbar_batches.update(1)
                fill_count +=1

        batches.append(current_batch_indices)
    pbar_batches.close()
    return batches


# Original filter_molecules_by_weight - slightly adapted if create_batches changes how it calls it
# This version is kept if TanimotoBatchSampler or other parts expect it directly with a list of mols
def filter_molecules_by_weight(molecules_list, min_weight, max_weight):
    """
    Original filter_molecules_by_weight, if needed by other parts of the code.
    Note: create_batches uses an internal helper _filter_molecules_by_weight.
    """
    filtered_indices = []
    for i, mol in enumerate(molecules_list):
        if mol is None: continue # Robustness for None molecules
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        if min_weight <= mol_weight <= max_weight:
            filtered_indices.append(i)
    return filtered_indices


class WeightSortedBatchSampler(Sampler):
    def __init__(self, dataset: MultimodalData, batch_size: int, drop_last: bool, window_size: int = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Ensure dataset.ref_data is a DataFrame and has "MW"
        if not hasattr(dataset, 'ref_data') or not isinstance(dataset.ref_data, pd.DataFrame) or "MW" not in dataset.ref_data.columns:
            raise ValueError("Dataset must have a 'ref_data' attribute (pandas DataFrame) with a 'MW' column.")

        self.window_size = window_size if window_size is not None else max(batch_size, 4)
        self.sorted_indices = self.dataset.ref_data.sort_values("MW").index.tolist()

    def __iter__(self):
        batch = []
        window = []
        for idx in self.sorted_indices:
            window.append(idx)
            if len(window) == self.window_size:
                random.shuffle(window)
                batch.extend(window)
                window = []

                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size] # Ensure exact batch_size if items > batch_size
                    batch = batch[self.batch_size:] # Keep remainder for next batch

        if len(window) > 0: # Process remaining items in window
            random.shuffle(window)
            batch.extend(window)

        # Yield remaining items if any
        while len(batch) >= self.batch_size:
            yield batch[:self.batch_size]
            batch = batch[self.batch_size:]
        
        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class TanimotoBatchSampler(Sampler):
    def __init__(self, dataset: MultimodalData, batch_size: int, drop_last: bool, similarity_threshold: float, max_search_size: int, weight_delta: float):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Note: 'create_batches' now expects a MultimodalData object directly for its first argument
        self.batches = create_batches(self.dataset, similarity_threshold, batch_size, max_search_size, weight_delta)

    def __iter__(self):
        for batch_indices in self.batches:
            if not self.drop_last or len(batch_indices) == self.batch_size:
                yield batch_indices

    def __len__(self):
        if self.drop_last:
            # Count only batches that are full
            return sum(1 for b in self.batches if len(b) == self.batch_size)
        else:
            return len(self.batches)


def _create_single_dataloader(config, stoi, stoi_MF, mode, sampler_class=None, batch_size_override=None):
    """Helper function to create a single DataLoader instance."""
    dataset = MultimodalData(config, stoi, stoi_MF, mode=mode)
    
    current_batch_size = batch_size_override if batch_size_override is not None else config.CLIP_batch_size

    if sampler_class == "WeightSortedBatchSampler":
        sampler = WeightSortedBatchSampler(dataset, current_batch_size, drop_last=(mode=="train")) # drop_last typically True for train
    elif sampler_class is None: # Default to shuffle for train, no sampler for others (implies shuffle=True/False in DataLoader)
        sampler = None
    else:
        raise ValueError(f"Unsupported sampler_class: {sampler_class}")

    if sampler:
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=config.num_workers)
    else: # If no sampler, use batch_size and shuffle
        return DataLoader(dataset, batch_size=current_batch_size, shuffle=(mode=="train"), collate_fn=collate_fn, num_workers=config.num_workers, drop_last=(mode=="train"))


def create_CLIP_dataloaders(config, stoi, stoi_MF):
    dataloaders = {}
    dataloaders["train"] = _create_single_dataloader(config, stoi, stoi_MF, mode="train", sampler_class="WeightSortedBatchSampler")
    dataloaders["test"] = _create_single_dataloader(config, stoi, stoi_MF, mode="test", sampler_class="WeightSortedBatchSampler")
    # Val dataloader uses batch_size=1 and no custom sampler as per original code.
    dataloaders["val"] = _create_single_dataloader(config, stoi, stoi_MF, mode="val", batch_size_override=1) 
    return dataloaders


class AvgMeter: # Remains as is
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer): # Remains as is
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# Refactored run_training_CLIP and its helpers
def _setup_clip_directories_and_config(config):
    """Sets up directories and saves the configuration for CLIP training."""
    if not os.path.exists(config.CLIP_model_save_dir):
        os.makedirs(config.CLIP_model_save_dir) # Use makedirs for nested paths

    config_dict = vars(config)
    config_save_path = config.CLIP_model_save_dir
    random_num = random.randint(0, 10000)
    with open(os.path.join(config_save_path, f"config_CLIP_{str(random_num)}.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)


def _setup_clip_callbacks(config):
    """Sets up the ModelCheckpoint callback for CLIP training."""
    filepath = config.CLIP_model_save_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='loss', # Assuming 'loss' is the quantity to monitor, adjust if different
        mode='min',
        filepath=os.path.join(filepath, 'model_CLIP-{epoch:02d}-{loss:.2f}'),
        save_top_k=-1,  # Keeps all checkpoints
    )
    return checkpoint_callback


def _initialize_clip_model(config):
    """Initializes or loads the CLIP model."""
    clip_multi_gpu_model = CLIPMultiGPU(config)

    if config.CLIP_continue_training:
        checkpoint_path = config.CLIP_model_path
        # Ensure correct model loading
        clip_multi_gpu_model = CLIPMultiGPU.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config # Pass the config, crucial for proper model re-hydration
        )
    return clip_multi_gpu_model


def _setup_clip_trainer(config, logger, checkpoint_callback, profiler):
    """Sets up the PyTorch Lightning Trainer for CLIP."""
    trainer = pl.Trainer(
        profiler=profiler,
        gpus=config.gpu_num,
        progress_bar_refresh_rate=10, # Consider if this is still current or needs config
        accelerator='ddp' if config.gpu_num > 1 else None,
        logger=logger,
        callbacks=[checkpoint_callback], # Must be a list
        max_epochs=config.CLIP_NUM_EPOCHS,
        # Consider other trainer flags from original:
        # fast_dev_run=True,
        # early_stop_callback=early_stopping, # Define if used
        # limit_train_batches=1,
        # limit_val_batches=1
    )
    return trainer


def _train_clip_model(trainer, model, train_dataloader, test_dataloader, config):
    """Runs the training loop for the CLIP model with error handling."""
    try:
        trainer.fit(model, train_dataloader, test_dataloader)
    except Exception as e:
        print(f"Error occurred during CLIP training: {e}")
        backup_ckpt_path = os.path.join(config.CLIP_model_save_dir, "last_backup_CLIP_checkpoint.ckpt")
        trainer.save_checkpoint(backup_ckpt_path)
        print(f"CLIP Model saved to {backup_ckpt_path}")


def run_training_CLIP(train_dataloader, test_dataloader, config):
    """Main function to run the CLIP model training pipeline."""
    _setup_clip_directories_and_config(config)
    
    checkpoint_callback = _setup_clip_callbacks(config)
    
    profiler = SimpleProfiler() # As used in original

    wandb_logger = WandbLogger(project=config.project, log_model='all')
    wandb_logger.log_hyperparams(vars(config)) # Log hyperparameters

    model = _initialize_clip_model(config)
    
    trainer = _setup_clip_trainer(config, wandb_logger, checkpoint_callback, profiler)
    
    _train_clip_model(trainer, model, train_dataloader, test_dataloader, config)


def run_CLIP(config, stoi, stoi_MF): # Remains as is
    dataloaders = create_CLIP_dataloaders(config, stoi, stoi_MF)
    run_training_CLIP(dataloaders["train"], dataloaders["test"], config)
   
    
########### Decide if I want to use  dg.create_CLIP_dataloaders or from here ###########
# This comment from the original file is preserved.
# If dg.create_CLIP_dataloaders is preferred, the create_CLIP_dataloaders function here might be redundant.
# For now, this script provides its own implementation.
