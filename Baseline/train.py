from sklearn.model_selection import train_test_split
from utils import shift_right, tokenize_data, collator, compute_metrics
from torch.utils.data import DataLoader
from dataset import TranslationDataset
from functools import partial


import pandas as pd


import transformers
import argparse
import evaluate
import torch

class CustomTrainer(transformers.Seq2SeqTrainer):
    '''
    A custom trainer class for the base model

    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        english_input_ids = inputs.get("input_ids")
        russian_input_ids = inputs.get("labels")
        english_attention_mask = inputs.get("attention_mask")
        russian_attention_mask = inputs.get("labels_attention_mask")
        
        en_ru = model(
            input_ids=english_input_ids,
            attention_mask=english_attention_mask,
            labels=russian_input_ids,
            decoder_attention_mask=shift_right(russian_attention_mask, pad_token_id=0),
        )

        ru_en = model(
            input_ids=russian_input_ids,
            attention_mask=russian_attention_mask,
            labels=english_input_ids,
            decoder_attention_mask=shift_right(english_attention_mask, pad_token_id=0),   
        )

        loss = ru_en.loss + en_ru.loss

        return (loss, {"ru_en": ru_en, "en_ru": en_ru}) if return_outputs else loss
    

if __name__ == '__main__':
    # Creating argument parser
    parser = argparse.ArgumentParser(
        description='Train script for the baseline model',
    )

    # Defining and parsing command line arguments
    parser.add_argument('-n', '--model_name', default='cointegrated/rut5-base', help='Model name for the output saving.')
    parser.add_argument('-b', '--batch_size', default=64, help='Batch size of the model.')
    parser.add_argument('-s', '--seed', default=42, help='Seed for the RNG.')
    parser.add_argument('-d', '--data', default='data/rus.txt', help='Path to the data csv.')

    args = parser.parse_args()

    # Reading data from CSV
    data = pd.read_csv(args.data, sep="\t", names=["en", "ru", "attribution"])

    # Train-test split
    trainval, test = train_test_split(data, test_size=0.2, random_state=args.seed)
    train, val = train_test_split(trainval, test_size=0.2, random_state=args.seed)

    # Tokenization of the splitted data
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name)
    tokenized = {
        "train": tokenize_data(train, tokenizer),
        "val": tokenize_data(val, tokenizer),
        "test": tokenize_data(test, tokenizer),
    }

    # Creating torch datasets
    train_ds = TranslationDataset(*tokenized["train"])
    val_ds = TranslationDataset(*tokenized["val"])
    test_ds = TranslationDataset(*tokenized["test"])

    # Creating dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # Selecting device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Creating arguments object for the trainer
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=f"./results/baseline-{args.model_name}",
        num_train_epochs=5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=5e-5,
        weight_decay=0.1,
        logging_steps=10,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        fp16=True, 
        #remove_unused_columns=False,
    )

    # Defining model
    model = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)

    # Creating optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate, 
        weight_decay=training_args.weight_decay
    )

    # Creating scheduler
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=opt, 
        num_warmup_steps=int(len(train_loader) * training_args.num_train_epochs // 5), 
        num_training_steps=int(len(train_loader) * training_args.num_train_epochs),
    )

    metric = partial(compute_metrics, tokenizer=tokenizer, metric=evaluate.load("sacrebleu"))

    trainer = transformers.Seq2SeqTrainer(
        model=model, 
        args=training_args, 
        data_collator=collator, 
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        optimizers=(opt, scheduler),
        compute_metrics=metric,
    )

    trainer.train()

    trainer.save_state()