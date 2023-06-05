#%%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from scipy import stats
from tqdm import tqdm
from IPython import display
from matplotlib_inline import backend_inline
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

#%% Utils for both models

def try_gpu(i=0):
    """Determine the device for training"""
    if torch.cuda.device_count() >= i+1:
        return torch.device("cuda", i)
    return torch.device("cpu")

def get_close_data():
    with open("../data/dataset_tensor/train/stocks.txt", "r") as file:
        stocks = [stock.split("\n")[0] for stock in file.readlines()]

    with open("../data/dataset_tensor/train/date.txt", "r") as file:
        dates_train = [date.split("\n")[0] for date in file.readlines()]

    with open("../data/dataset_tensor/eval/date.txt", "r") as file:
        dates_eval = [date.split("\n")[0] for date in file.readlines()]

    with open("../data/dataset_tensor/test/date.txt", "r") as file:
        dates_test = [date.split("\n")[0] for date in file.readlines()]

    dates = dates_train + dates_test + dates_eval
    close_list = [pd.read_csv(f"../data/dataset_per_stocks/{stock}.csv", index_col = 1).loc[dates, "close"].sort_index() for stock in stocks]
    close = pd.concat(close_list, axis = 1)
    close.columns = stocks
    close.to_csv("../backtest/close.csv")
    return close

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Function to set axes for Animator"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
class Animator(object):
    """Animator used for showing the loss while training"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, 
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), 
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait = True)

#%% Utils for factorVAE

def get_dataset(type):
    """Get training, evaluating and testing dataset"""
    features_dataset = torch.load(f"../data/dataset_tensor/{type}/feat.pt")
    returns_dataset  = torch.load(f"../data/dataset_tensor/{type}/ret.pt")
    print(f"Total step: {features_dataset.shape[0]}")
    print(f"Time span: {features_dataset.shape[1]}")
    print(f"Stock size: {features_dataset.shape[2]}")
    print(f"Feature size: {features_dataset.shape[3]}")
    return features_dataset, returns_dataset

def get_dataloader_factorVAE(features, label, device, batch_size, shuffle=True):
    """Fetch dataloader which is used for training factorVAE model"""
    features = torch.Tensor(features).to(device)
    label = torch.Tensor(label).to(device)
    dataset = TensorDataset(features, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def calculate_rankIC(model, features, returns, sampler_num=200, repeat=10):
    """Calculate rank-IC"""
    sampler = list(RandomSampler(range(features.shape[0]), num_samples=sampler_num)) # Randomly select dataset to calculate rankIC 
    features_selected = features[sampler, :]
    returns_selected = returns[sampler, :]
    returns_prediction = torch.cat([model.prediction(features[:200, :])[0] for _ in range(repeat)], dim = -1).mean(dim = -1)
    rankIC = np.mean([stats.spearmanr(returns_prediction[i, :], returns_selected[i, :])[0] for i in range(returns_prediction.shape[0])])
    return rankIC

def train_factorVAE(dataloader, model, optimizer, epochs, features_train, returns_train, features_eval, returns_eval, repeat = 5):
    """Training factorVAE"""
    animator = Animator(xlabel="epochs", xlim=[0, epochs], legend=["Train RankIC", "Eval RankIC"])
    for epoch in range(epochs):
        print(f"=== Epoch: {epoch} ===")
        for batch, (feat, ret) in enumerate(dataloader):
            loss = model.run_model(feat, ret)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                rankIC_train = calculate_rankIC(model, features_train, returns_train, repeat = repeat)
                rankIC_eval = calculate_rankIC(model, features_eval, returns_eval, repeat = repeat)
                animator.add(epoch + batch / len(dataloader), (rankIC_train, rankIC_eval))
    return model

def get_weights(model, features, type, stock_num = 20, repeat = 5):
    "Generate stock weights for investment"
    # Fetch the dates and stocks list
    with open(f"../data/dataset_tensor/{type}/date.txt", "r") as file:
        dates = [date.split("\n")[0] for date in file.readlines()]
    with open(f"../data/dataset_tensor/{type}/stocks.txt", "r") as file:
        stocks = [stock.split("\n")[0] for stock in file.readlines()]
    # Calculate the weights based on predictions
    y_pred = sum([model.prediction(features)[0].squeeze(-1) for _ in tqdm(range(repeat))]) / repeat
    weights = (y_pred >= y_pred.sort(descending=True)[0][:, stock_num-1:stock_num]).to(torch.float32).numpy() / stock_num
    weights = np.where(weights == 0, np.nan, weights)
    weights = pd.DataFrame(index=dates, columns=stocks, data = weights)
    weights = weights.shift(1).dropna(how="all", axis=0) # shift 1 day because of the cap between prediction and backtest
    return weights


#%% Utils for time series model

def dataset_dim_convertion_4to3(features, returns, reduction = True):
    """Convert the dim of dataset from 4 to 2"""
    time_span, features_num = features.shape[1], features.shape[3] 
    return (features.permute(0, 2, 1, 3).reshape(-1, time_span, features_num),
            returns.reshape(-1, 1))

def grad_clipping(net, theta):  #@save
    """Gradient clipping for LSTM"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def get_weights_ts(model, test_dl, type, stock_nums = 20):
    """Generate stock weight for investment"""
    # Fetch the dates and stocks list
    with torch.no_grad():
        y_pred_all = 0
        for batch, (X, y) in tqdm(enumerate(test_dl)):
            y_pred = model(X)
            if batch == 0:
                y_pred_all = y_pred.clone()
                continue
            y_pred_all = torch.concat([y_pred_all, y_pred], dim=0)
    y_pred_all = y_pred_all.reshape(-1, 100)
    torch.cuda.empty_cache()
    with open(f"../data/dataset_tensor/{type}/date.txt", "r") as file:
            dates = [date.split("\n")[0] for date in file.readlines()]
    with open(f"../data/dataset_tensor/{type}/stocks.txt", "r") as file:
        stocks = [stock.split("\n")[0] for stock in file.readlines()]
    # Calculate the weights based on predictions
    weights = (y_pred_all >= y_pred_all.sort(descending=True)[0][:, stock_nums-1:stock_nums]).to(torch.float32).cpu().numpy() / stock_nums
    weights = np.where(weights == 0, np.nan, weights)
    weights = pd.DataFrame(index=dates, columns=stocks, data = weights)
    weights = weights.shift(1).dropna(how="all", axis=0) # shift 1 day because of the cap between prediction and backtest
    y_pred_df = pd.DataFrame(index=dates, columns=stocks, data=y_pred_all.numpy())
    y_pred_df = y_pred_df.shift(2).dropna() # shift 2 day because of the cap between prediction and actual rate
    return weights

def train_modelTS(dataloader, model, optimizer, loss, device, epochs, features_eval, returns_eval, model_name, eval_sample_size):
    """Training time series models including LSTM Transformer and PatchTST"""
    animator = Animator(xlabel="epochs", xlim=[0, epochs], legend=["MSE train", "MSE eval"])
    print("sadasdsadsadas")
    for epoch in range(epochs):
        print(f"====epoch:{epoch}====")
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            mse_train = loss(y, model(X))
            optimizer.zero_grad()
            mse_train.backward()
            if model_name == "LSTM": 
                grad_clipping(model, 1) # Gradient clipping for LSTM
            optimizer.step()
            if (batch % 100 == 0) and (batch > 0):
                with torch.no_grad():
                    eval_sample = list(RandomSampler(range(features_eval.shape[0]), num_samples=eval_sample_size))
                    mse_eval = loss(model(features_eval[eval_sample, :].to(device)), returns_eval[eval_sample, :].to(device))
                animator.add(epoch + batch / len(dataloader), (mse_train.item(), mse_eval.item()))
    return model