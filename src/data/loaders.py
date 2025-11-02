from torch.utils.data import DataLoader
from datasets import load_from_disk
from .dataset import GeneformerDataset
from .sampler import LengthGroupedSampler, collate_fn_dynamic_pad


def create_dataloaders(dataset_path, batch_size=12, shuffle=True, num_workers=4, pad_token_id=0, val_split=0.05, seed=42):
    
    raw_dataset = load_from_disk(dataset_path)
    split = raw_dataset.train_test_split(test_size=val_split, seed=seed)
        
    train_dataset = GeneformerDataset(split['train'])
    val_dataset = GeneformerDataset(split['test'])
        
    train_sampler = LengthGroupedSampler(train_dataset.get_lengths(), batch_size, shuffle=True)
    val_sampler = LengthGroupedSampler(val_dataset.get_lengths(), batch_size, shuffle=False)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn_dynamic_pad(b, pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
        )
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=lambda b: collate_fn_dynamic_pad(b, pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
        )
        
    return train_loader, val_loader