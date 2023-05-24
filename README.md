# Quantitative Investment Based on Transformer

A quantitative investment project based on `FactorVAE`, `PatchTST`, `Transformer` and `LSTM`.

Our goal is to build a stock selection startegy which can achieve excess alpha.

---

## Repository Structure

- backtest: weights and results from prediction and backtest.
- data: features, labels, dates and stocks of training, evaluating and testing dataset.
- models: the repository to store the optimal model including `FactorVAE`, `PatchTST`, `Transformer` and  `LSTM`.
- reference: reference of this project.
- src: source code of the project.

---

## Source Code Structure

- `Models`: Package including the code of all the models.

- `utils.py`: Utilities

- `backtest.py`: The code of backtest system.

- `dataset_generation.ipynb`: The code to preprocess the raw data and generate training, evaluating and testing dataset.

- `model_train.ipynb`: Train and save models.
  
  The figure of training loss is included in the notebook which is convenient to take a glance.

- `model_backtest.ipynb`: Generate the weight for investment and backtest.

  The figure of backtest net value is included in the notebook which is convenient to take a glance. 

---

## Dataset Structure

The dataset in the zip file contains four parts:

- `feat.pt`: `torch.tensor`
  
  This dataset is the features based on `alpha158`
  
  Shape of the dataset: (S, T, N, F), the implication of each parameter is as follows:
  
  S: number of samples
  
  T: time window for each sample
  
  N: number of stocks 
  
  F: the number of features

- `ret.pt`: `torch.tensor`
  
  This dataset is the daily return rate of each stocks and maps the `feat.pt`.
  
  Shape of the dataset: (S, N), the implication of each paramerter is the same as `feat.pt`.

- `date.txt`: `List`

  The trading date of the dataset. The dataset segmentation is as follows:
  
  training dataset: `2010-01-01` to `2017-12-31`
  
  evaluating dataset: `2018-01-01` to `2019-12-31`
  
  testing dataset: `2020-01-01` to `2021-12-31`

- `stocks.txt`: `List`

  The stock list of the project, which is same between different repository.

---

