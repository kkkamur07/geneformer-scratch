import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import os


from src.model.student_model import StudentModel
from src.model.teacher_model import TeacherModel
from src.data.dataset import GeneformerDataset
from src.data.collator import GeneDataCollator

def calculate_metrics(logits, labels):
    
    mask = labels != -100
    
    if mask.sum() == 0:
        return None, None
    
    active_logits = logits[mask] 
    active_labels = labels[mask] 
    
    # --- Metric 1: Accuracy ---
    preds = torch.argmax(active_logits, dim=-1)
    acc = (preds == active_labels).float().mean().item()
    
    # --- Metric 2: Perplexity ---
    loss = F.cross_entropy(active_logits, active_labels)
    ppl = torch.exp(loss).item()
    
    return acc, ppl

@hydra.main(version_base=None, config_path="/home/krrish/Desktop/Programming/geneformer-scratch/configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    full_dataset = GeneformerDataset(cfg.data.val_dataset_path)
    
    indices = np.random.choice(len(full_dataset), 100, replace=False)
    small_dataset = Subset(full_dataset, indices)
    
    collator = GeneDataCollator(
        mlm_probability=0.15 
    )
    
    loader = DataLoader(small_dataset, batch_size=16, collate_fn=collator)


    print("Loading Teacher...")
    teacher = TeacherModel(
        model_path=cfg.teacher.model_path,
        device=device
    )


    print("Loading Student...")
    student = StudentModel(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        device=device
    )
    
    # Load the best checkpoint
    ckpt_path = "/home/krrish/Desktop/Programming/geneformer-scratch/outputs/checkpoints_/geneformer_distillation/model_best.pt"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            student.load_state_dict(checkpoint['model_state_dict'])
        elif "model" in checkpoint:
            student.load_state_dict(checkpoint["model"])
        else:
            student.load_state_dict(checkpoint)
            
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}. Using random weights!")
    
    student.model.eval() 

    teacher_metrics = {"acc": [], "ppl": []}
    student_metrics = {"acc": [], "ppl": []}

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Teacher Forward
            t_logits = teacher(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_logits=True
            )

            # Student Forward
            s_logits = student(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_logits=True
            )
            
            # Calculate Metrics
            t_acc, t_ppl = calculate_metrics(t_logits, labels)
            s_acc, s_ppl = calculate_metrics(s_logits, labels)

            if t_acc is not None:
                teacher_metrics["acc"].append(t_acc)
                teacher_metrics["ppl"].append(t_ppl)
                student_metrics["acc"].append(s_acc)
                student_metrics["ppl"].append(s_ppl)


    print("\n" + "="*60)
    print("="*60)
    print(f"{'Metric':<15} | {'Teacher (Target)':<18} | {'Student (Yours)':<18} | {'Gap':<10}")
    
    t_acc_avg = np.mean(teacher_metrics['acc'])
    s_acc_avg = np.mean(student_metrics['acc'])
    
    acc_gap = s_acc_avg - t_acc_avg
    print(f"{'MLM Accuracy':<15} | {t_acc_avg:.4f}             | {s_acc_avg:.4f}             | {acc_gap:+.4f}")
    
    t_ppl_avg = np.mean(teacher_metrics['ppl'])
    s_ppl_avg = np.mean(student_metrics['ppl'])
    ppl_gap = s_ppl_avg - t_ppl_avg
    
    print(f"{'Perplexity':<15} | {t_ppl_avg:.2f}              | {s_ppl_avg:.2f}              | {ppl_gap:+.2f}")
    print("="*60)

if __name__ == "__main__":
    main()