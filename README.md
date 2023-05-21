# CycleTranslate

Neural Machine Translation with cycle-consistency loss.

Final project on Skoltech DL-2023 course.

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

Create environment with
```
micromamba create -f env.yml
```

## Team:
- Seleznyov Mikhail
- Sushko Nikita
- Kovaleva Maria
