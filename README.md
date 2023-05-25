# CycleTranslate

Neural Machine Translation with cycle-consistency loss.

Final project on Skoltech DL-2023 course.

## Problem definition 

Traditional methods of Neural Machine Translation (NMT) require a big corpus of paired texts. However, in our information age, we have a lot of unstructured unpaired data in the internet.
So, can we employ this unpaired data in a smart way to train an NMT language model?

The idea behind our method is simple -- we can train one model to translate text from one language to another and another model to translate the translation back to the original
language. Since the result after the cyclic translation shold be at least really close or even the same, we can effectively treat unlabeled data as labeled, eliciting more signal from what
we have.

Thus, these questions arise:
- Can we benefit from additional unlabeled data?
- Can we benefit from enforcing cycle consistency in low-data regime?
- Does it depend on the model size?

## Main results

The first experiment was to train four t5-base models on a small subset of labeled data for 10 epochs. First two models were to be trained using the classic CrossEntropy loss, while two 
other models employed our proposed Cyclic Loss.

| Experiment name | BLEU for en2ru model | BLEU for ru2en model |
|---|---|---|
| T5-base, CrossEntropy loss, 10 epochs, subset of labeled data | 4.7844 | 4.3415 |
| T5-base, Cyclic Loss, 10 epochs, subset of labeled data | **6.1359** | **5.7527** |

As you can see, on low number of epochs, proposed method of training yields much better BLEU scores.

The next experiment was to again train four t5-base models on a small subset of labeled data, but this time do the training for 30 epochs instead of 10 epochs. Again, first two models were
to be trained using the classic CrossEntropy loss, while two other models employed our proposed Cyclic Loss.

| Experiment name | BLEU for en2ru model | BLEU for ru2en model |
|---|---|---|
| T5-base, CrossEntropy loss, 30 epochs, subset of labeled data | **14.7513** | **18.7511** |
| T5-base, Cyclic Loss, 30 epochs, subset of labeled data | 13.7197 | 17.6676 |

As you can see, on higher amount of epochs, our models yield a tad lower scores, but still comparable with the classical approach.

The third experiment was to train six t5-base models, using different amount of data. First four models use only the small subset of labeled data that was used in previous experiments and 
CrossEntropy Loss, while the last two models use Cyclic Loss and are being trained in the following manner:
- Train for 10 epochs on the subset of labeled data
- Train for 1 epoch on a big set of unlabeled data
- Train for 10 epochs on the subset of labeled data

| Experiment name | BLEU for en2ru model | BLEU for ru2en model |
|---|---|---|
| T5-base, CrossEntropy loss, 10 epochs, subset of labeled data | 4.7844 | 4.3415 |
| T5-base, CrossEntropy loss, 30 epochs, subset of labeled data | **14.7513** | **18.7511** |
| T5-base, Cyclic Loss, 21 epochs, multistage | 6.7953 | 7.2948 |

Our approach showed much worse results than longer training on labeled data, but better results than training for 10 epochs.

In conclusion, Cycle Consistency loss can help when training models for a short time. In other cases it does not provide any benefit and only worsens the score.

## Getting data

```
wget http://www.manythings.org/anki/rus-eng.zip

mkdir data
mv rus-eng.zip data
cd data
unzip rus-eng.zip
```

## Setup

[Install](https://mamba.readthedocs.io/en/latest/installation.html) `micromamba`. For Linux users:
```
curl micro.mamba.pm/install.sh | bash
```
Restart the terminal.

Create and activate environment with
```
micromamba create -f env.yml
micromamba activate cyc
```

Configure PYTHOPATH:
```
cd path/to/CycleTranslate
export PROJECT_DIR=$(pwd)
micromamba env config vars set PYTHONPATH=${PROJECT_DIR}:

micromamba deactivate
micromamba activate
```

Add new kernel for your notebooks:
```
python -m ipykernel install --user --name cyc --display-name "Python (cyc)"
```

Then your can run `jupyter` or `jupyterlab`, select corresponding kernel and start working.

## Team:
- Seleznyov Mikhail - implemented the baseline model, ran experiments, worked on the presentation, experiment design
- Sushko Nikita - wrote helper and utility scripts, ran experiments, wrote this summary in the README, idea of the project
- Kovaleva Maria - implemented the cyclic loss model, worked on the presentation
