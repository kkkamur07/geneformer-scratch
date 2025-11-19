import torch.nn as nn
import torch
from transformers import BertForMaskedLM


class TeacherModel(nn.Module):
    def __init__(
        self, 
        model_path,
        device: torch.device = torch.device("cpu")
        ):
        
        super().__init__()
        self.device = device
        self.model = BertForMaskedLM.from_pretrained(model_path).to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode
        self.model.eval()
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        return_logits=True
        ):

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device), 
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None, 
                labels=labels.to(self.device) if labels is not None else None,
                output_hidden_states=True,
            )
        
        if return_logits:
            return outputs.logits 
        else:
            return outputs
        
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)