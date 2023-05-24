import os
import argparse

import pandas as pd
import numpy as np

import torch
import transformers
import evaluate
from torch.utils.data import DataLoader
from functools import partial
from tqdm.auto import tqdm, trange

from common import shift_right, tokenize_data, collator, compute_metrics
from common import TranslationDataset


def train_one_epoch(model, dataloader, optimizer, scheduler, flip_direction, device="cuda:0"):
    model.train()
    
    losses = []
    
    for batch in tqdm(dataloader, desc="train"):
        optimizer.zero_grad()

        source_language_input_ids = batch["input_ids"].to(device)
        target_language_input_ids = batch["labels"].to(device)
        source_language_attention_mask = batch["attention_mask"].to(device)
        target_language_attention_mask = batch["labels_attention_mask"].to(device)
        
        if flip_direction:
            source_language_input_ids, target_language_input_ids = target_language_input_ids, source_language_input_ids
            source_language_attention_mask, target_language_attention_mask = target_language_attention_mask, source_language_attention_mask

        out = model(
            input_ids=source_language_input_ids,
            attention_mask=source_language_attention_mask,
            labels=target_language_input_ids,
        )
        loss = out.loss
        
        loss.backward()

        losses.append(loss.item())
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

    metrics = {
        "total_loss": sum(losses) / len(losses),
    }
    
    return losses, metrics


@torch.no_grad()
def eval_model(model, dataloader, metric, num_beams, max_new_tokens, flip_direction, device="cuda:0"):
    model.eval()
    
    all_labels = []
    all_preds = []
    
    for batch in tqdm(dataloader, desc="val"):
        source_language_input_ids = batch["input_ids"].to(device)
        target_language_input_ids = batch["labels"].to(device)
        source_language_attention_mask = batch["attention_mask"].to(device)
        target_language_attention_mask = batch["labels_attention_mask"].to(device)
        
        if flip_direction:
            source_language_input_ids, target_language_input_ids = target_language_input_ids, source_language_input_ids
            source_language_attention_mask, target_language_attention_mask = target_language_attention_mask, source_language_attention_mask

        out = model.generate(
            input_ids=source_language_input_ids,
            attention_mask=source_language_attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        all_labels.append(target_language_input_ids.detach().cpu())
        all_preds.append(out[0].detach().cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    return metric((all_preds, all_labels))


if __name__ == "__main__":
    # Creating argument parser
    parser = argparse.ArgumentParser(
        description="Train script for the baseline model",
    )

    # Defining and parsing command line arguments
    parser.add_argument("-n", "--model_name", default="cointegrated/rut5-base", help="Model name to load from huggingface.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size of the model.")
    parser.add_argument("-s", "--seed", default=42, help="Seed for the RNG.")
    parser.add_argument("-d", "--data", default="data", help="Path to the data folder.")
    parser.add_argument("--run-name", help="Name of the experiment.")

    args = parser.parse_args()

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(f"results/{args.run_name}"):
        os.mkdir(f"results/{args.run_name}")

    # Reading data
    print("Reading data...")
    train = pd.read_csv(f"{args.data}/low_resource_train.tsv", sep="\t", names=["en", "ru"], index_col=0)
    val = pd.read_csv(f"{args.data}/val.tsv", sep="\t", names=["en", "ru"], index_col=0)
    test = pd.read_csv(f"{args.data}/test.tsv", sep="\t", names=["en", "ru"], index_col=0)

    # Tokenization of the splitted data
    print("Tokenizing...")
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name)
    tokenized_data = {
        "train": tokenize_data(train, tokenizer),
        "val": tokenize_data(val, tokenizer),
        "test": tokenize_data(test, tokenizer),
    }

    # Creating torch datasets
    train_ds = TranslationDataset(*tokenized_data["train"])
    val_ds = TranslationDataset(*tokenized_data["val"])
    test_ds = TranslationDataset(*tokenized_data["test"])

    # Creating dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Selecting device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = dict(
        n_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.0,
    )

    metric = partial(compute_metrics, tokenizer=tokenizer, metric=evaluate.load("sacrebleu"))

    for flip_direction in (False, True):

        # Defining model
        model = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)

        # Creating optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        # Creating scheduler
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=opt, 
            num_warmup_steps=int(len(train_loader) * config["n_epochs"] // 5), 
            num_training_steps=int(len(train_loader) * config["n_epochs"]),
        )

        print("Starting train...")

        all_losses = []
        all_metrics = []
        for epoch in trange(config["n_epochs"], desc="epoch"):
            losses, _ = train_one_epoch(model, train_loader, opt, scheduler, flip_direction, device)
            
            all_losses.extend(losses)

            prefix = "ru2en" if flip_direction else "en2ru"
            torch.save(model.state_dict(), f"results/{args.run_name}/{prefix}_model.pth")
            np.save(f"results/{args.run_name}/{prefix}_losses.npy", all_losses)
        
        metrics = eval_model(model, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=flip_direction, device=device)
        print(f"{prefix} validation:", metrics)
        all_metrics.append(metrics)
        pd.DataFrame(all_metrics).to_csv(f"results/{args.run_name}/{prefix}_val_metrics.csv")

    print("Success!")