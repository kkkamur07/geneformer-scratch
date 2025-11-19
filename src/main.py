import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.model.teacher_model import TeacherModel
from src.model.student_model import StudentModel
from src.data.dataset import GeneformerDataset
from src.training.trainer import DistillationTrainer
from src.training.logging import TrainingLogger
from torch.utils.data import DataLoader
from torch.optim import AdamW


@hydra.main(version_base=None, config_path="/home/krrish/Desktop/Programming/geneformer-scratch/configs", config_name="config")
def main(cfg: DictConfig):
    
    # Print config
    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Device
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=cfg.paths.log_dir,
        experiment_name=cfg.names.experiment_name
    )
    
    logger.info("🔧 Loading models...")
    
    # Load teacher model
    teacher = TeacherModel(model_path=cfg.paths.teacher_model_path)
    logger.info(f"Teacher model loaded from {cfg.paths.teacher_model_path} with {teacher.get_num_parameters():,} parameters")
    
    # Create student model
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
    
    generator = torch.Generator().manual_seed(cfg.seed)
    
    logger.info(f"Student model created with parameters:{student.get_num_parameters():,} total, {student.get_trainable_parameters():,} trainable")
    
    torch.compile(student)
    logger.info(f"Student model compiled with torch.compile()")
    
    #! There is something fundamentally wrong here.
    # # Load dataset
    # logger.info("Loading dataset...")
    # dataset = GeneformerDataset(cfg.data.dataset_path)
    
    # # Split into train/val
    # train_size = int(len(dataset) * cfg.data.train_split)
    # val_size = len(dataset) - train_size
    
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size]
    # )
    
    # logger.info(f"Train size: {len(train_dataset):,}")
    # logger.info(f"Val size: {len(val_dataset):,}")
    
    # logger.info("Creating Samplers")

    # all_lengths = dataset.lengths()
    # logger.info("Obtained sequence lengths from the dataset")
    # train_lengths = all_lengths[train_dataset.indices].tolist()
    # val_lengths = all_lengths[val_dataset.indices].tolist()
    
    # train_sampler = LengthGroupedSampler(
    #     batch_size=cfg.data.batch_size,
    #     dataset=train_dataset,
    #     lengths=train_lengths,
    #     generator=generator
    # )
    
    # val_sampler = LengthGroupedSampler(
    #     batch_size=cfg.data.batch_size,
    #     dataset=val_dataset,
    #     lengths=val_lengths,
    #     generator=None
    # )

    
    # # Create dataloaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     sampler=train_sampler,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=cfg.data.pin_memory,
    #     prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     sampler=val_sampler,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=cfg.data.pin_memory,
    #     prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
    # )
    
    logger.info(f"DataLoaders created")
    
    # Optimizer
    optimizer = AdamW(
        student.parameters(),
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        logger=logger
    )
    
    trainer.train()
    
    # Close logger
    logger.close()
    
    print("Training complete!")


if __name__ == "__main__":
    main()