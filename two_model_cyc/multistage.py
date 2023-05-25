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


def train_one_epoch(model_en2ru, model_ru2en, dataloader, 
                    optimizer_en2ru, optimizer_ru2en, 
                    scheduler_en2ru, scheduler_ru2en,
                    num_beams, max_new_tokens, device="cuda:0", treat_as_unlabeled=False):
    model_en2ru.train()
    model_ru2en.train()
    
    losses = []
    
    for batch in tqdm(dataloader, desc="train"):
        optimizer_en2ru.zero_grad()
        optimizer_ru2en.zero_grad()

        english_input_ids = batch["input_ids"].to(device) #eng
        russian_input_ids = batch["labels"].to(device) #ru
        english_attention_mask = batch["attention_mask"].to(device)
        russian_attention_mask = batch["labels_attention_mask"].to(device)

        eng2ru_generated = model_en2ru.generate(input_ids = english_input_ids, 
                                           attention_mask = english_attention_mask,
                                           num_beams=num_beams,
                                           max_new_tokens=max_new_tokens,)
        
        ru2eng_generated = model_ru2en.generate(input_ids = russian_input_ids, 
                                           attention_mask = russian_attention_mask,
                                           num_beams=num_beams,
                                           max_new_tokens=max_new_tokens,)
        
        # Cycle Consistency loss
        # (eng -> rus) -> eng
        en_cyc_loss = model_ru2en(
            input_ids = eng2ru_generated,
            attention_mask = torch.ones_like(eng2ru_generated),
            labels = english_input_ids,
        ).loss

        # (rus -> eng) -> rus
        ru_cyc_loss = model_en2ru(
            input_ids = ru2eng_generated,
            attention_mask = torch.ones_like(ru2eng_generated),
            labels = russian_input_ids,
        ).loss

        if treat_as_unlabeled:
            en_ru_loss = 0
            ru_en_loss = 0
        else:
            en_ru_loss = model_en2ru(
                input_ids = english_input_ids,
                attention_mask = english_attention_mask,
                labels = russian_input_ids,
            ).loss

            ru_en_loss = model_ru2en(
                input_ids = russian_input_ids,
                attention_mask = russian_attention_mask,
                labels = english_input_ids,
            ).loss

        loss = ru_en_loss + en_ru_loss + 0.1 * (en_cyc_loss + ru_cyc_loss)
        
        loss.backward()

        losses.append(loss.item())
        
        optimizer_en2ru.step()
        optimizer_ru2en.step()

        if scheduler_en2ru is not None:
            scheduler_en2ru.step()
        if scheduler_ru2en is not None:
            scheduler_ru2en.step()

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
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size of the model.")
    parser.add_argument("-s", "--seed", default=42, help="Seed for the RNG.")
    parser.add_argument("-d", "--data", default="data", help="Path to the data folder.")
    parser.add_argument("--run-name", help="Name of the experiment.")
    parser.add_argument("--en2ru-checkpoint", default=None, help="ru2en model to start with.")
    parser.add_argument("--ru2en-checkpoint", default=None, help="en2ru model to start with.")

    args = parser.parse_args()

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(f"results/{args.run_name}"):
        os.mkdir(f"results/{args.run_name}")

    # Reading data
    print("Reading data...")
    train = pd.read_csv(f"{args.data}/low_resource_train.tsv", sep="\t", names=["en", "ru"], index_col=0)
    unlabeled_train = pd.read_csv(f"{args.data}/remaining_train.tsv", sep="\t", names=["en", "ru"], index_col=0)
    val = pd.read_csv(f"{args.data}/val.tsv", sep="\t", names=["en", "ru"], index_col=0)
    test = pd.read_csv(f"{args.data}/test.tsv", sep="\t", names=["en", "ru"], index_col=0)

    # Tokenization of the splitted data
    print("Tokenizing...")
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name)
    tokenized_data = {
        "train": tokenize_data(train, tokenizer),
        #"unlabeled_train": tokenize_data(train, tokenizer),
        #"val": tokenize_data(train, tokenizer),
        #"test": tokenize_data(train, tokenizer),
        "unlabeled_train": tokenize_data(unlabeled_train, tokenizer),
        "val": tokenize_data(val, tokenizer),
        "test": tokenize_data(test, tokenizer),
    }

    # Creating torch datasets
    train_ds = TranslationDataset(*tokenized_data["train"])
    unlabeled_train_ds = TranslationDataset(*tokenized_data["unlabeled_train"])
    val_ds = TranslationDataset(*tokenized_data["val"])
    test_ds = TranslationDataset(*tokenized_data["test"])

    # Creating dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    unlabeled_train_loader = DataLoader(unlabeled_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Selecting device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = dict(
        n_init_epochs=0,
        n_cyc_epochs=1,
        n_finetune_epochs=5,
        learning_rate=3e-4,
        weight_decay=0.0,
    )

    metric = partial(compute_metrics, tokenizer=tokenizer, metric=evaluate.load("sacrebleu"))

    # Defining models
    model_en2ru = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)
    model_ru2en = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)

    if args.en2ru_checkpoint is not None:
        print(f"Loading checkpoint from {args.en2ru_checkpoint}")
        model_en2ru.load_state_dict(torch.load(args.en2ru_checkpoint))
    if args.ru2en_checkpoint is not None:
        print(f"Loading checkpoint from {args.ru2en_checkpoint}")
        model_ru2en.load_state_dict(torch.load(args.ru2en_checkpoint))

    # Creating optimizers
    opt_en2ru = torch.optim.AdamW(model_en2ru.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    opt_ru2en = torch.optim.AdamW(model_ru2en.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Creating schedulers
    n_steps = (config["n_init_epochs"] + config["n_finetune_epochs"]) * len(train_loader) + \
        config["n_cyc_epochs"] * len(unlabeled_train_loader)

    scheduler_en2ru = transformers.get_cosine_schedule_with_warmup(optimizer=opt_en2ru, 
        num_warmup_steps=int(n_steps // 5), 
        num_training_steps=n_steps,
    )
    scheduler_ru2en = transformers.get_cosine_schedule_with_warmup(optimizer=opt_ru2en, 
        num_warmup_steps=int(n_steps // 5), 
        num_training_steps=n_steps,
    )

    print("Starting train...")

    all_losses = []
    all_metrics_en2ru = []
    all_metrics_ru2en = []

# Train models to make them at least bit sensible
    for epoch in trange(config["n_init_epochs"], desc="epoch"):
        losses, _ = train_one_epoch(model_en2ru, model_ru2en, train_loader, 
                                    opt_en2ru, opt_ru2en, 
                                    scheduler_en2ru, scheduler_ru2en,
                                    num_beams=1, max_new_tokens=36, device=device)
        all_losses.extend(losses)

        torch.save(model_en2ru.state_dict(), f"results/{args.run_name}/en2ru_model_init.pth")
        torch.save(model_ru2en.state_dict(), f"results/{args.run_name}/ru2en_model_init.pth")
        np.save(f"results/{args.run_name}/both_losses.npy", all_losses)
    
    # metrics_en2ru = eval_model(model_en2ru, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=False, device=device)
    # metrics_ru2en = eval_model(model_ru2en, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=True, device=device)

    # print(f"en2ru init validation:", metrics_en2ru)
    # print(f"ru2en init validation:", metrics_ru2en)

    # all_metrics_en2ru.append(metrics_en2ru)
    # all_metrics_ru2en.append(metrics_ru2en)

    # pd.DataFrame(all_metrics_en2ru).to_csv(f"results/{args.run_name}/en2ru_val_metrics.csv")
    # pd.DataFrame(all_metrics_ru2en).to_csv(f"results/{args.run_name}/ru2en_val_metrics.csv")

# Employ cyclic loss on big dataset
    for epoch in trange(config["n_cyc_epochs"], desc="epoch"):
        losses, _ = train_one_epoch(model_en2ru, model_ru2en, unlabeled_train_loader, 
                                    opt_en2ru, opt_ru2en, 
                                    scheduler_en2ru, scheduler_ru2en,
                                    num_beams=1, max_new_tokens=36, device=device, treat_as_unlabeled=True)
        all_losses.extend(losses)

        torch.save(model_en2ru.state_dict(), f"results/{args.run_name}/en2ru_model_cyc.pth")
        torch.save(model_ru2en.state_dict(), f"results/{args.run_name}/ru2en_model_cyc.pth")
        np.save(f"results/{args.run_name}/both_losses.npy", all_losses)
    
    metrics_en2ru = eval_model(model_en2ru, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=False, device=device)
    metrics_ru2en = eval_model(model_ru2en, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=True, device=device)

    print(f"en2ru cyc validation:", metrics_en2ru)
    print(f"ru2en cyc validation:", metrics_ru2en)

    all_metrics_en2ru.append(metrics_en2ru)
    all_metrics_ru2en.append(metrics_ru2en)

    pd.DataFrame(all_metrics_en2ru).to_csv(f"results/{args.run_name}/en2ru_val_metrics.csv")
    pd.DataFrame(all_metrics_ru2en).to_csv(f"results/{args.run_name}/ru2en_val_metrics.csv")

# Tune models with labeled dataset
    for epoch in trange(config["n_finetune_epochs"], desc="epoch"):
        losses, _ = train_one_epoch(model_en2ru, model_ru2en, train_loader, 
                                    opt_en2ru, opt_ru2en, 
                                    scheduler_en2ru, scheduler_ru2en,
                                    num_beams=1, max_new_tokens=36, device=device)
        all_losses.extend(losses)

        torch.save(model_en2ru.state_dict(), f"results/{args.run_name}/en2ru_model_finetune.pth")
        torch.save(model_ru2en.state_dict(), f"results/{args.run_name}/ru2en_model_finetune.pth")
        np.save(f"results/{args.run_name}/both_losses.npy", all_losses)
    
    metrics_en2ru = eval_model(model_en2ru, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=False, device=device)
    metrics_ru2en = eval_model(model_ru2en, val_loader, metric, num_beams=2, max_new_tokens=36, flip_direction=True, device=device)

    print(f"en2ru finetune validation:", metrics_en2ru)
    print(f"ru2en finetune validation:", metrics_ru2en)

    all_metrics_en2ru.append(metrics_en2ru)
    all_metrics_ru2en.append(metrics_ru2en)

    pd.DataFrame(all_metrics_en2ru).to_csv(f"results/{args.run_name}/en2ru_val_metrics.csv")
    pd.DataFrame(all_metrics_ru2en).to_csv(f"results/{args.run_name}/ru2en_val_metrics.csv")

    print("Success!")