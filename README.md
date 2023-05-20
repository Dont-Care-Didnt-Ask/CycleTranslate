# CycleTranslate

Neural Machine Translation with cycle-consistency loss.

Final project on Skoltech DL-2023 course.

## Getting data

```
wget http://www.manythings.org/anki/rus-eng.zip
```

## Setup

```
conda create -n cycle pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate cycle
conda install --file requirements.txt -c huggingface
```

## Team:
- Seleznyov Mikhail
- Sushko Nikita
- Kovaleva Maria