import torch
from torch.utils.data import DataLoader, Sampler
import random
import os


class LengthGroupedSampler(Sampler):
    """
    Groups samples by similar lengths to minimize padding waste.
    """
    def __init__(self, lengths, batch_size, shuffle=True):

        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Step 1: Create (index, length) pairs
        index_length_pairs = [(i, length) for i, length in enumerate(lengths)]
        
        # Step 2: Sort by length (this actually sorts the indices)
        index_length_pairs.sort(key=lambda x: x[1])
        
        # Step 3: Extract sorted indices
        self.sorted_indices = [idx for idx, _ in index_length_pairs]
        
    def __iter__(self):
   
        batches = [
            self.sorted_indices[i:i + self.batch_size] 
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        if self.shuffle:
            random.shuffle(batches)
        
        # Flatten batches back to indices
        for batch in batches:
            yield from batch
    
    def __len__(self):
        return len(self.sorted_indices)


def collate_fn_dynamic_pad(batch, pad_token_id=0):

    max_len = max(item['length'] for item in batch)
    
    input_ids = []
    attention_masks = []
    
    for item in batch:
        seq = item['input_ids']  # List of token IDs
        seq_len = item['length']
        
        # Calculate padding needed for THIS sequence
        padding_len = max_len - seq_len
        
        # Pad input_ids: [original tokens] + [pad_token] * padding_len
        padded_seq = seq + [pad_token_id] * padding_len
        
        # Creates attention mask
        mask = [1] * seq_len + [0] * padding_len
        
        input_ids.append(padded_seq)
        attention_masks.append(mask)
    
    # Convert to tensors
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.int16),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.bool),
    }