from torch.utils.data import Dataset


import torch


class TranslationDataset(Dataset):
    '''
    Dataset for the base model

    '''
    def __init__(self, english, russian):
        super().__init__()
        self.english = english
        self.russian = russian
        
        assert english is not None or russian is not None
    
    def __getitem__(self, index):
        item = {}

        if self.english is not None:
            item["input_ids"] = torch.tensor(self.english["input_ids"][index], dtype=torch.long)
            item["attention_mask"] = torch.tensor(self.english["attention_mask"][index])
        
        if self.russian is not None:
            item["labels"] = torch.tensor(self.russian["input_ids"][index], dtype=torch.long)
            item["labels_attention_mask"] = torch.tensor(self.russian["attention_mask"][index])

        assert len(item) > 0
        
        return item

    def __len__(self):
        return len(self.english["input_ids"]) if self.english is not None else len(self.russian["input_ids"])