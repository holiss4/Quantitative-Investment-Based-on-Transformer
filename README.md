# Quantitative Investment Based on Transformer

A quantitative investment project based on `FactorVAE` which is derived from the `Transformer`.

---

## Repository Structure

- backtest: weights and results from prediction and backtest.
- data: features, labels, dates and stocks of training, evaluating and testing dataset.
- reference: reference of this project.
- src: source code of the project.

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
  
  training dataset: `2010-01-01` to `2014-12-30`
  
  evaluating dataset: `2015-01-01` to `2016-12-31`
  
  testing dataset: `2017-01-01` to `2021-12-31`

- `stocks.txt`: `List`

  The stock list of the project, which is same between different repository
