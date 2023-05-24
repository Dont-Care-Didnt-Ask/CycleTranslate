from utils import shift_right, tokenize_data, collator, compute_metrics
from torch.utils.data import DataLoader
from dataset import TranslationDataset
from functools import partial


import pandas as pd


import transformers
import argparse
import evaluate
import torch

if __name__ == '__main__':
    # Creating argument parser
    parser = argparse.ArgumentParser(
        description='Demo script for the baseline model',
    )

    # Defining and parsing command line arguments
    parser.add_argument('-n', '--model_name', default='cointegrated/rut5-base', help='Model name for the output saving.')
    parser.add_argument('-s', '--seed', default=42, help='Seed for the RNG.')
    parser.add_argument('-d', '--data', default='data', help='Path to the data folder.')
    parser.add_argument('-p', '--checkpoint_path', help='Path to the checkpoint.')

    args = parser.parse_args()

    # Reading data
    print("Reading data...")
    train = pd.read_csv(f"{args.data}/low_resource_train.tsv", sep="\t", names=["en", "ru"], index_col=0)
    val = pd.read_csv(f"{args.data}/val.tsv", sep="\t", names=["en", "ru"], index_col=0)
    test = pd.read_csv(f"{args.data}/test.tsv", sep="\t", names=["en", "ru"], index_col=0)

    # Tokenization of the splitted data
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name)

    # Selecting device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Loading model...")
    model = transformers.T5ForConditionalGeneration(transformers.T5Config.from_pretrained(args.model_name)).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)

    metric = partial(compute_metrics, tokenizer=tokenizer, metric=evaluate.load("sacrebleu"))

    examples = test.sample(5, random_state=args.seed)
    
    for i in range(len(examples)):
        en = examples.iloc[i]["en"]
        ru = examples.iloc[i]["ru"]

        en_tok = tokenizer([en], return_tensors="pt")
        ru_tok = tokenizer([ru], return_tensors="pt")

        en_tok = {k: v.to(device) for k, v in en_tok.items()}
        ru_tok = {k: v.to(device) for k, v in ru_tok.items()}

        ru_translate_tok = model.generate(**en_tok, num_beams=8, max_new_tokens=30)
        en_translate_tok = model.generate(**ru_tok, num_beams=8, max_new_tokens=30)

        ru_translate = tokenizer.batch_decode(ru_translate_tok)
        en_translate = tokenizer.batch_decode(en_translate_tok)

        print("En, ground truth: ", en)
        print("Ru, ground truth: ", ru)
        print("Ru, translate:", ru_translate)
        print("En, translate:", en_translate)
        print()