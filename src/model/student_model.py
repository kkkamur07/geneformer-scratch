import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig


class StudentModel(nn.Module):
    def __init__(
        self,
        vocab_size=25426,
        hidden_size=256,
        num_hidden_layers=6, 
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        device : torch.device = torch.device("cpu")
        ):
        super().__init__()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

        config._attn_implementation = "sdpa"
        self.device = device
        self.model = BertForMaskedLM(config).to(self.device)
        self.config = config
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_logits=True):
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            output_hidden_states=False, 
        )
        
        if return_logits:
            return outputs.logits  # [batch_size, seq_len, vocab_size]
        else:
            return outputs
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)