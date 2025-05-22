# Standard library imports
import ast
import pickle
import random
import re
import threading # Note: threading is imported but not explicitly used.
import os
from typing import List, Dict, Any, Tuple, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.nn import functional as F # Used in collate_fn
from torch.utils.data import Dataset
from tqdm.auto import tqdm 
from torch.nn.utils.rnn import pad_sequence

# Local application/library specific imports
# Ensure these paths are correct relative to where this file will be used.
# If helper_functions_pl_v15_4 and smi_augmenter_v15_4 are also moved to utils_MMST,
# these imports might need to be `from . import ...`
import utils_MMT.helper_functions_pl_v15_4 as hf 
from utils_MMT.smi_augmenter_v15_4 import SMILESAugmenter

tqdm.pandas()

# --- SmilesEnumerator Class ---
class SmilesEnumerator(object):
    def __init__(self, charset: str = '@C)(=cOn1S2/H[N]\\', pad: int = 120, 
                 leftpad: bool = True, isomericSmiles: bool = True, 
                 enum: bool = True, canonical: bool = False):
        self._charset: Optional[str] = None
        self.charset: str = charset # Uses setter
        self.pad: int = pad
        self.leftpad: bool = leftpad
        self.isomericSmiles: bool = isomericSmiles
        self.enumerate: bool = enum # 'enum' is a python keyword, consider renaming variable
        self.canonical: bool = canonical
        self._char_to_int: Dict[str, int] = {}
        self._int_to_char: Dict[int, str] = {}
        self._charlen: int = 0

    @property
    def charset(self) -> Optional[str]:
        return self._charset
        
    @charset.setter
    def charset(self, charset_val: str) -> None:
        self._charset = charset_val
        self._charlen = len(charset_val)
        self._char_to_int = {c: i for i, c in enumerate(charset_val)}
        self._int_to_char = {i: c for i, c in enumerate(charset_val)}
        
    def fit(self, smiles: Union[np.ndarray, pd.Series], extra_chars: Optional[List[str]] = None, extra_pad: int = 5) -> None:
        if extra_chars is None: extra_chars = []
        charset_set = set("".join(list(smiles)))
        self.charset = "".join(charset_set.union(set(extra_chars))) # Uses setter
        self.pad = max([len(s) for s in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles: str) -> Optional[str]:
        m = Chem.MolFromSmiles(smiles)
        if m is None: return None
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles_array: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((smiles_array.shape[0], self.pad, self._charlen), dtype=np.int8)
        for i, ss in enumerate(smiles_array):
            if self.enumerate:
                ss_random = self.randomize_smiles(ss)
                ss = ss_random if ss_random is not None else ss # Fallback to original if randomization fails
            l = len(ss)
            start_index = self.pad - l if self.leftpad else 0
            for j, char_val in enumerate(ss):
                if char_val in self._char_to_int: # Check if char is in charset
                    one_hot[i, start_index + j, self._char_to_int[char_val]] = 1
        return one_hot
      
    def reverse_transform(self, vect: np.ndarray) -> np.ndarray:       
        smiles_list: List[str] = []
        for v_single in vect:
            v_filtered = v_single[v_single.sum(axis=1) == 1] # Valid one-hot rows
            smile = "".join(self._int_to_char[i] for i in v_filtered.argmax(axis=1))
            smiles_list.append(smile)
        return np.array(smiles_list)  

# --- HSQC Data Pollution ---
def pollute_HSQC_data(
    hsqc_df: pd.DataFrame, 
    noise_peaks: List[List[Union[float, int]]], 
    noise_num_list: List[int]
) -> pd.DataFrame:
    if not noise_num_list: return hsqc_df # No numbers of peaks to choose from
    num_peaks_to_add = random.choice(noise_num_list)
    if num_peaks_to_add == 0: return hsqc_df
    
    selected_noise_peaks = random.sample(noise_peaks, min(num_peaks_to_add, len(noise_peaks)))
    new_rows: List[Dict[str, Union[float, int]]] = []
    for peak in selected_noise_peaks:
        x, y, _ = peak # Assuming intensity from noise_peaks is not used for new direction
        direction = 1 if np.random.random() < 0.5 else -1
        new_rows.append({"F2 (ppm)": x, "F1 (ppm)": y, 'direction': direction})
    
    if new_rows: # Only concat if there are new rows to add
        return pd.concat([hsqc_df, pd.DataFrame(new_rows)], ignore_index=True)
    return hsqc_df

# --- MultimodalData Dataset Class ---
class MultimodalData(Dataset):
    def __init__(self, config: Any, stoi: Dict[str, int], stoi_MF: Dict[str, int], mode: str):
        self.config = config
        self.stoi = stoi
        self.stoi_MF = stoi_MF
        self.mode = mode 
        self.smiles_randomizer = SMILESAugmenter(restricted=True)
        self.fingerprint_size = config.fingerprint_size
        
        self.spectrum_configs = {
            "1H": {"dims": 2, "norm_fn": lambda x: [[s[0]/10.0, s[1]] for s in x if isinstance(s, list) and len(s)==2]},
            "13C": {"dims": 1, "norm_fn": lambda x: [s/200.0 for s in x if isinstance(s, (int, float))]},
            "HSQC": {"dims": 2, "norm_fn": lambda x: self._normalize_2d_shifts(x, "HSQC")},
            "COSY": {"dims": 2, "norm_fn": lambda x: self._normalize_2d_shifts(x, "COSY")},
            "IR": {"dims": 1, "is_special_loader": True} 
        }
        self.possible_datasets = list(self.spectrum_configs.keys())
        
        self.data_dict: Dict[str, Any] = {} 
        self.ref_data: pd.DataFrame = pd.DataFrame() 

        if self.config.data_type == "sgnn": 
            self._load_data_for_mode()
        else:
            raise ValueError(f"Unsupported data_type in config: {self.config.data_type}")

    def _calculate_min_max_mw(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'SMILES' in df.columns:
            df_copy = df.copy() 
            df_copy.loc[:, 'MW'] = df_copy['SMILES'].progress_apply(hf.calculate_mol_weight)
            self.config.train_weight_min = df_copy['MW'].min()
            self.config.train_weight_max = df_copy['MW'].max()
            return df_copy
        return df

    def _reshuffle_and_save_pickle(self, dict_of_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        print("Processing data to save as pkl file...")
        reshuffled_dict: Dict[str, Dict[str, Any]] = {}
        all_sample_ids = set()
        sample_id_to_smiles: Dict[str, str] = {}

        for df_content in dict_of_dfs.values(): 
            for sample_id, row in df_content.iterrows(): 
                sample_id_str = str(sample_id) 
                all_sample_ids.add(sample_id_str) 
                if 'SMILES' in row and sample_id_str not in sample_id_to_smiles:
                    sample_id_to_smiles[sample_id_str] = row['SMILES']
        
        for sample_id_str in tqdm(all_sample_ids, desc="Reshuffling data for pickle"):
            reshuffled_dict[sample_id_str] = {key: [] for key in self.possible_datasets} 
            reshuffled_dict[sample_id_str]['SMILES'] = sample_id_to_smiles.get(sample_id_str, "")
            for spec_type, df_content in dict_of_dfs.items():
                if sample_id_str in df_content.index.map(str): 
                    row_data_series = df_content.loc[df_content.index.map(str) == sample_id_str]
                    if not row_data_series.empty:
                        shift_data = row_data_series.iloc[0].get('shifts')
                        if shift_data is not None:
                            reshuffled_dict[sample_id_str][spec_type] = [shift_data] # Store as list of one item
        
        pickle_base_path = getattr(self.config, "pickle_base_path", "./") 
        if not os.path.exists(pickle_base_path): os.makedirs(pickle_base_path)
        # Use a more deterministic pickle name
        pickle_filename = f"processed_multimodal_data_{self.mode}_{self.config.data_size}.pkl" 
        new_file_path = os.path.join(pickle_base_path, pickle_filename)
        
        self.config.current_pickle_file_path = new_file_path # Save path for potential reuse
        
        print(f"Saving processed data to pickle: {new_file_path}")
        with open(new_file_path, 'wb') as f: pickle.dump(reshuffled_dict, f)
        print("Pickle saved.")
        return reshuffled_dict

    def _load_data_for_mode(self) -> None:
        config_paths = {
            "1H": self.config.csv_1H_path_SGNN, "13C": self.config.csv_13C_path_SGNN,
            "HSQC": self.config.csv_HSQC_path_SGNN, "COSY": self.config.csv_COSY_path_SGNN,
            "IR": self.config.csv_IR_MF_path
        }
        pickle_base_path = getattr(self.config, "pickle_base_path", "./")
        # Use a more deterministic pickle name, including data_size in name
        expected_pickle_path = os.path.join(pickle_base_path, f"processed_multimodal_data_{self.mode}_{self.config.data_size}.pkl")
        
        load_pickle_path = getattr(self.config, "current_pickle_file_path", expected_pickle_path)
        if not os.path.exists(load_pickle_path) and os.path.exists(expected_pickle_path) : # check expected if current not found
            load_pickle_path = expected_pickle_path

        if os.path.exists(load_pickle_path):
            print(f"Loading data from existing pickle: {load_pickle_path}")
            with open(load_pickle_path, 'rb') as f: self.data_dict = pickle.load(f)
        else:
            print(f"Pickle not found at {load_pickle_path} (or {expected_pickle_path}). Loading from CSVs...")
            raw_data_dfs: Dict[str, pd.DataFrame] = {}
            for spec_type in self.possible_datasets: # Iterate over defined possible_datasets
                if spec_type in self.config.training_mode: # training_mode should list active spectra
                    csv_path = config_paths.get(spec_type)
                    if csv_path and os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            if 'Unnamed: 0' in df.columns: df = df.drop('Unnamed: 0', axis=1)
                            if 'sample-id' in df.columns: df.set_index('sample-id', inplace=True)
                            else: print(f"Warning: 'sample-id' missing in {csv_path}. Using index.")
                            raw_data_dfs[spec_type] = df
                        except Exception as e: print(f"Error loading CSV {spec_type} from {csv_path}: {e}")
                    else: print(f"Warning: CSV path for {spec_type} not found or not specified for active spectrum.")
            if not raw_data_dfs: raise FileNotFoundError("No raw data CSVs loaded for active spectra in training_mode.")
            self.data_dict = self._reshuffle_and_save_pickle(raw_data_dfs)

        ref_path_key = self.config.ref_data_type # e.g. "1H", this CSV defines the items in dataset
        ref_csv_path = config_paths.get(ref_path_key) if self.mode != "val" else self.config.csv_path_val
        
        if not (ref_csv_path and os.path.exists(ref_csv_path)): 
            raise FileNotFoundError(f"{self.mode} ref CSV for '{ref_path_key if self.mode != 'val' else 'val'}' not found: {ref_csv_path}")
        
        self.ref_data = pd.read_csv(ref_csv_path)
        if self.config.data_size > 0 and len(self.ref_data) > self.config.data_size : 
             self.ref_data = self.ref_data.iloc[:self.config.data_size]
        
        if self.mode != "val": self.ref_data = self._calculate_min_max_mw(self.ref_data)
        if self.mode in ["train", "test"]: self.ref_data = self._split_data_for_mode(self.ref_data)
        
        if 'sample-id' not in self.ref_data.columns:
            if self.ref_data.index.name == 'sample-id': self.ref_data.reset_index(inplace=True)
            else: self.ref_data['sample-id'] = self.ref_data.index # Fallback to index if no 'sample-id'
        self.ref_data['sample-id'] = self.ref_data['sample-id'].astype(str) # Ensure sample_id is string


    def _zero_pad_tensor(self, data_list: List[Any], pad_length: int, item_dims: int = 1, 
                         inner_dim_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]: 
        # Mask: True for padded elements, False for valid data
        if not isinstance(data_list, list): data_list = [] 

        if item_dims == 1:
            current_len = len(data_list)
            if current_len == 0: return self._create_empty_tensor_and_mask(pad_length, item_dims, inner_dim_size)
            try: data_tensor = torch.tensor(data_list, dtype=torch.float)
            except Exception: return self._create_empty_tensor_and_mask(pad_length, item_dims, inner_dim_size)

            if current_len >= pad_length:
                return data_tensor[:pad_length], torch.zeros(pad_length, dtype=torch.bool) 
            else:
                padding = torch.zeros(pad_length - current_len, dtype=torch.float)
                mask = torch.ones(pad_length, dtype=torch.bool); mask[:current_len] = False 
                return torch.cat((data_tensor, padding)), mask
        elif item_dims == 2:
            current_len = len(data_list)
            if current_len == 0: return self._create_empty_tensor_and_mask(pad_length, item_dims, inner_dim_size)
            try: data_tensor = torch.tensor(data_list, dtype=torch.float)
            except Exception: return self._create_empty_tensor_and_mask(pad_length, item_dims, inner_dim_size)
            
            actual_inner_dim = data_tensor.shape[1] if data_tensor.ndim == 2 and data_tensor.shape[1] > 0 else inner_dim_size
            if data_tensor.ndim == 1 : # Became 1D (e.g. list of floats, not list of lists)
                return self._create_empty_tensor_and_mask(pad_length, item_dims, inner_dim_size)

            if current_len >= pad_length:
                return data_tensor[:pad_length, :actual_inner_dim], torch.zeros(pad_length, dtype=torch.bool)
            else:
                padding = torch.zeros(pad_length - current_len, actual_inner_dim, dtype=torch.float)
                mask = torch.ones(pad_length, dtype=torch.bool); mask[:current_len] = False
                return torch.cat((data_tensor, padding)), mask
        else: raise ValueError(f"Unsupported item_dims: {item_dims}")

    def _split_data_for_mode(self, data_df: pd.DataFrame) -> pd.DataFrame:
        split_idx = int(self.config.tr_te_split * len(data_df))
        if self.mode == "train": return data_df.iloc[:split_idx].reset_index(drop=True)
        elif self.mode == "test": return data_df.iloc[split_idx:].reset_index(drop=True)
        return data_df 

    def _load_ir_spectrum(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = os.path.join(self.config.IR_data_folder, f"{sample_id}.csv")
        try:
            df_ir = pd.read_csv(file_path)
            spectra_list = df_ir['spectra'].tolist()
            if not spectra_list: return self._create_empty_tensor_and_mask(self.config.input_dim_IR, 1)
            max_val = float(max(spectra_list)) if max(spectra_list) > 0 else 1.0
            binned_spectrum = np.zeros(self.config.input_dim_IR)
            if spectra_list: 
                avg_span = len(spectra_list) / float(self.config.input_dim_IR)
                curr_pos = 0.0
                for i in range(self.config.input_dim_IR):
                    s, e = round(curr_pos), round(curr_pos + avg_span)
                    s = min(s, len(spectra_list) -1); e = min(e, len(spectra_list))
                    if s >= e: binned_spectrum[i] = spectra_list[s] / max_val if len(spectra_list) > s else 0.0
                    else: binned_spectrum[i] = np.mean(spectra_list[s:e]) / max_val
                    curr_pos += avg_span
            return torch.tensor(binned_spectrum, dtype=torch.float), torch.zeros(self.config.input_dim_IR, dtype=torch.bool)
        except Exception: return self._create_empty_tensor_and_mask(self.config.input_dim_IR, 1)

    def _normalize_2d_shifts(self, shifts_2d: List[List[float]], spec_type: str) -> List[List[float]]:
        # Normalizes 2D NMR shifts (HSQC, COSY).
        div_f2, div_f1 = (10.0, 200.0) if spec_type == "HSQC" else (10.0, 10.0) # COSY uses 10,10
        return [[p[0] / div_f2, p[1] / div_f1] for p in shifts_2d if isinstance(p, list) and len(p) == 2]

    def _create_empty_tensor_and_mask(self, pad_length: int, item_dims: int = 1, inner_dim_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        # Utility to create zero tensor and all-True mask (indicating all padding).
        shape = (pad_length,) if item_dims == 1 else (pad_length, inner_dim_size)
        return torch.zeros(shape, dtype=torch.float), torch.ones(pad_length, dtype=torch.bool) # Mask True for padded

    def _encode_text_data(self, text: str, stoi_map: Dict[str, int], max_len: int, is_mf: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encodes SMILES or MF string, adds SOS/EOS, and pads.
        if is_mf:
            pattern = r'([A-Z][a-z]*)(\d{1,2})?' # MF tokenization
            tokens = [t for item_tuple in re.findall(pattern, text) for t in item_tuple if t] # flatten and remove empty
        else: # SMILES
            tokens = hf.tokenize_smiles(text, hf.two_char_symbols) # Assumes hf has these defined
        
        encoded = [stoi_map.get(token, stoi_map.get('<UNK>', 0)) for token in tokens]
        encoded_w_sos_eos = [stoi_map.get("<SOS>", 0)] + encoded + [stoi_map.get("<EOS>",
