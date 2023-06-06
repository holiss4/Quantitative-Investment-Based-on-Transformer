#%%
import os
import torch
import backtest as bt
from utils import *
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

class metric(object):
    """Calculate the metrics of the strategy and model"""
    def __init__(self, ret_pred):
        self.ret_pred = ret_pred
        self.ret_pred.index = pd.to_datetime(self.ret_pred.index)
        self.close = pd.read_csv("../backtest/close.csv", index_col=0, parse_dates=True, header=0)
        self.ret_true = self.close.pct_change().loc[self.ret_pred.index, :]
        self.metric_dict = dict()
    
    def metric_accuracy(self):
        """Calculate the accuracy of the returns prediction"""
        accuracy = ((self.ret_true > 0) == (self.ret_pred > 0)).mean()
        self.metric_dict["accuracy"] = accuracy
        return accuracy
    
def model_backtest(model_name, mode = "eval", gen_weight = True, device = "cuda:0", batch_size=128):
    """Weight generation and backtest for models"""
    assert model_name in ["FactorVAE", "Transformer", "LSTM", "PatchTST"], "The model name is wrong, try 'FactorVAE', 'Transformer', 'LSTM'"
    # Get training weight
    weight_path = f"../backtest/{model_name}/{mode}/weights.csv"
    y_pred_path = f"../backtest/{model_name}/{mode}/returns_pred.csv"
    if (not os.path.exists(weight_path)) or gen_weight:
        print("===== Generating the weight =====")
        model = torch.load(f"../models/{model_name}.pth")  # load the model 
        model = model.to(device)
        if model_name == "FactorVAE":
            features, returns = get_dataset(mode)  # load the dataset
            test_dl = get_dataloader_factorVAE(features, returns, device, batch_size, shuffle=False)
            weights, y_pred = get_weights(model, test_dl, type=mode, repeat=5)
        if model_name in ["Transformer", "LSTM", "PatchTST"]:
            features, returns = dataset_dim_convertion_4to3(*get_dataset(mode))
            test_dl = get_dataloader_factorVAE(features, returns, device, batch_size, shuffle=False)
            weights, y_pred = get_weights_ts(model, test_dl, type=mode, stock_nums=20)
        weights.to_csv(weight_path)
        y_pred.to_csv(y_pred_path)
    if not gen_weight:
        y_pred = pd.read_csv(y_pred_path, index_col=0, parse_dates=True)
    # Params for backtest
    with open(f"../data/dataset_tensor/{mode}/date.txt", "r") as file:
        dates = [date.split("\n")[0] for date in file.readlines()]
    param = dict()
    param["init_value"] = 1
    param["start_date"] = dates[1] # Shift 1 day because of the cap between prediction and backtest
    param["end_date"] = dates[-1]
    param["threshold"] = 1
    param["days_back"] = 20
    param["fee"] = True
    param["stock_price"] = "../backtest/close.csv"  # close data
    param["stock_weight"] = f"../backtest/{model_name}/{mode}/weights.csv"      # weight of strategy
    param["result_savepath"] = f"../backtest/{model_name}/{mode}/result.xlsx"   # result save path
    param["figure_savepath"] = f"../backtest/{model_name}/{mode}/net_value.png" # figure save path
    # Backtest
    backtest_result = bt.backtest(param)
    backtest_result["returns_pred"] = y_pred
    # calculate metrics
    metrics = metric(backtest_result["returns_pred"])
    metrics.metric_accuracy()
    return backtest_result, metrics.metric_dict
    
model_name = "Transformer"

if model_name == "Transformer": backtest_result_Transformer, metrics = model_backtest("Transformer", gen_weight=True, device="cpu", mode="train")
if model_name == "FactorVAE": backtest_result_FactorVAE, metrics = model_backtest("FactorVAE", gen_weight=True, device="cpu", mode="test")
if model_name == "PatchTST": backtest_result_PatchTST, metrics = model_backtest("PatchTST", gen_weight=True)
if model_name == "LSTM": backtest_result_Transformer, metrics = model_backtest("LSTM")
