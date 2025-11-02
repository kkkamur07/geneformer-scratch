from torch.utils.data import Dataset
from datasets import load_from_disk


class GeneformerDataset(Dataset):
    def __init__(self, dataset):
        if isinstance(dataset, str):
            self.dataset = load_from_disk(dataset)
        else:
            self.dataset = dataset
        
        self.lengths = self.dataset['length']
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.dataset[idx]['input_ids'],
            'length': self.dataset[idx]['length']
        }
    
    def get_lengths(self):
        return self.lengths