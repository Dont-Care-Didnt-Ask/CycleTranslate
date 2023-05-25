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
        all_preds.append(out.detach().cpu())

    max_len = max(label.shape[1] for label in all_labels)
    all_labels = torch.cat([torch.nn.functional.pad(label, (0, max_len - label.shape[1]), value=-100) for label in all_labels]).numpy()

    max_len = max(pred.shape[1] for pred in all_preds)
    all_preds = torch.cat([torch.nn.functional.pad(pred, (0, max_len - pred.shape[1]), value=0) for pred in all_preds]).numpy()

    return metric((all_preds, all_labels))


if __name__ == "__main__":
    # Creating argument parser
    parser = argparse.ArgumentParser(
        description="Train script for the baseline model",
    )

    # Defining and parsing command line arguments
    parser.add_argument("-n", "--model_name", default="cointegrated/rut5-base", help="Model name to load from huggingface.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size of the model.")
    parser.add_argument("-rc", "--ru_checkpoint_path", default="results/None/en2ru_model.pth", help="Path to checkpoint for the RU model.")
    parser.add_argument("-ec", "--en_checkpoint_path", default="results/None/ruen_model.pth", help="Path to checkpoint for the EN model.")
    parser.add_argument("-d", "--data", default="data", help="Path to the data folder.")
    parser.add_argument("--run-name", default='eval_run', help="Name of the experiment.")

    args = parser.parse_args()

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

    # Defining metric
    metric = partial(compute_metrics, tokenizer=tokenizer, metric=evaluate.load("sacrebleu"))

    # Defining models
    model_en2ru = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)
    model_ru2en = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)

    # Loading checkpoints
    model_en2ru.load_state_dict(torch.load(args.en_checkpoint_path))
    model_ru2en.load_state_dict(torch.load(args.ru_checkpoint_path))

    # Evaluating models
    metrics_en2ru = eval_model(model_en2ru, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=False, device=device)
    metrics_ru2en = eval_model(model_ru2en, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=True, device=device)

    print(f"en2ru validation:", metrics_en2ru)
    print(f"ru2en validation:", metrics_ru2en)

    # Saving metrics to dataframe
    all_metrics_en2ru = [metrics_en2ru]
    all_metrics_ru2en = [metrics_ru2en]

    pd.DataFrame(all_metrics_en2ru).to_csv(f"results/{args.run_name}/en2ru_val_metrics.csv")
    pd.DataFrame(all_metrics_ru2en).to_csv(f"results/{args.run_name}/ru2en_val_metrics.csv")

    print("Success!")