import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def cat_and_pad_last_dim_if_needed(tensor_sequence, cat_dim=0, pad_value=0):
    max_len = max(ten.shape[-1] for ten in tensor_sequence)
    cat_tensor = torch.cat(
        [torch.nn.functional.pad(ten, (0, max_len - ten.shape[1]), value=pad_value) for ten in tensor_sequence],
        dim=cat_dim,
    )

    return cat_tensor


def tokenize_data(data, tokenizer):
    '''
    Method for data tokenization

    '''
    english_tokenized = None if "en" not in data else tokenizer(data["en"].tolist(), truncation=True, max_length=36)
    russian_tokenized = None if "ru" not in data else tokenizer(data["ru"].tolist(), truncation=True, max_length=36)

    return english_tokenized, russian_tokenized


def collator(examples):
    '''
    Method for batching keys

    '''
    batch_keys = examples[0].keys()

    batch = {key: pad_sequence([sample[key] for sample in examples], batch_first=True, padding_value=0)
            for key in batch_keys}

    return batch


def shift_right(tensor, pad_token_id):
    '''
    Method for teacher forcing
    
        See also: https://en.wikipedia.org/wiki/Teacher_forcing

    '''
    shifted_tensor = tensor.new_zeros(tensor.shape)
    shifted_tensor[..., 1:] = tensor[..., :-1].clone()
    shifted_tensor[..., 0] = pad_token_id

    return shifted_tensor


def postprocess_text(preds, labels):
    '''
    Method for stripping labels and predictions

    '''
    preds = [pred.strip().replace("_", " ") for pred in preds]
    labels = [[label.strip().replace("_", " ")] for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, metric):
    '''
    Method for computing metrics for predictions on evaluation stage

    '''
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    print("Some samples from compute_metric function:")
    for i in range(0, len(decoded_preds), len(decoded_preds) // 6):
        print("Pred:", decoded_preds[i])
        print("Label:", decoded_labels[i])
        print()

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result

