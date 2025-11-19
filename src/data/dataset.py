from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np
import torch

#! There is something fundamentally wrong here.
class GeneformerDataset(Dataset):
    def __init__(self, dataset_path_or_obj, indices=None, lengths=None):
        # Load dataset only if it's a path
        if isinstance(dataset_path_or_obj, str):
            self.dataset = load_from_disk(dataset_path_or_obj)
        else:
            self.dataset = dataset_path_or_obj

        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)  # Ensure correct dtype
        else:
            self.indices = np.arange(len(self.dataset), dtype=np.int64)

        if lengths is not None:
            self.lengths = np.asarray(lengths, dtype=np.int32)  # Explicit dtype
        else:
            # More efficient: access once and index
            self.lengths = np.array(self.dataset['length'], dtype=np.int32)[self.indices]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]  # No need for int() conversion
        item = self.dataset[real_idx]
        
        return {
            'input_ids': item['input_ids'],
            'length': item['length']
        }
    
    def get_lengths(self):
        return self.lengths

    def split(self, split_ratio=0.9, seed=42):
        total_size = len(self)
        train_size = int(total_size * split_ratio)
        
        # Create shuffled indices
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(self.indices)
        
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]

        # More efficient: slice pre-computed lengths instead of re-accessing dataset
        shuffled_lengths = self.lengths[rng.permutation(len(self.lengths))]
        train_lengths = shuffled_lengths[:train_size]
        val_lengths = shuffled_lengths[train_size:]

        # Create new dataset objects
        train_ds = GeneformerDataset(self.dataset, indices=train_indices, lengths=train_lengths)
        val_ds = GeneformerDataset(self.dataset, indices=val_indices, lengths=val_lengths)
        
        return train_ds, val_ds