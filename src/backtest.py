#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
def init_param():
    param = dict()
    param["init_value"] = 1
    param["start_date"] = "2010-05-05"
    param["end_date"] = "2014-12-30" 
    param["threshold"] = 1
    param["days_back"] = 20
    param["fee"] = True
    param["stock_price"] = "../backtest/close.csv"  # 股票池收盘价数据
    param["stock_weight"] = "../backtest/FactorVAE/train/weights.csv" # 策略权重
    param["result_savepath"] = "../backtest/FactorVAE/train/result.xlsx" # 结果保存路径
    return param

def backtest(param):
    # 初始化
    fee_in = 0.000
    fee_out = 0.000
    fee_man = 0.000
    tmp_fee_daily = 0.0

    weight = pd.read_csv(param["stock_weight"], index_col = 0, parse_dates = True)
    weight = weight.loc[param["start_date"]:param["end_date"]]
    weight.fillna(0, inplace = True)
    close_price = pd.read_csv(param["stock_price"], index_col = 0, parse_dates = True)
    close_price = close_price.loc[param["start_date"]:param["end_date"], weight.columns]
    close_rate = close_price.pct_change(axis = 0)

    # 设置股票权重, 持仓, 费用
    weight_real = pd.DataFrame(index = close_price.index, columns = close_price.columns.to_list() + ["cash"])
    share_hold = pd.DataFrame(index = close_price.index, columns = close_price.columns.to_list() + ["cash"])
    fee_detail = pd.DataFrame(index = weight.index, columns = ["current_fee", "accrued_fee"])
    fee_daily = pd.DataFrame(index = close_price.index, columns = ["management_fee"])
    net_position_after_man = pd.DataFrame(index = close_price.index, columns = ["net_value_after_management_fee"])

    # 获取换仓日列表
    date_list = weight.index.to_list()
    date_list = list(map(lambda x: str(x).split()[0], date_list))

    # 日频计算持仓和权重
    value_lag1 = param["init_value"]
    for i, date in enumerate(date_list):
        
        # 这里row是日频的
        row = np.where(weight_real.index == date)[0][0]
        weight_real.iloc[row, :-1], weight_real.iloc[row, -1] = weight.iloc[i], 1 - weight.iloc[i].sum()
        share_hold.iloc[row, :-1], share_hold.iloc[row, -1] = weight.iloc[i] * value_lag1, weight_real.iloc[row, -1] * value_lag1
        
        # 计算费用
        if param["fee"]:
            if i == 0:
                fee_detail.loc[date, "current_fee"] = param["init_value"] * fee_in
            else:
                share_change = share_hold.iloc[row, :-1] - share_hold.iloc[row-1, :-1]
                fee_buy = np.where(share_change > 0, share_change, 0).sum() * fee_in
                fee_sold = np.where(share_change < 0, share_change, 0).sum() * fee_out * (-1)
                fee = fee_buy + fee_sold + tmp_fee_daily
            
                change_ratio = (share_hold.iloc[row].sum() - fee) / share_hold.iloc[row].sum()
                share_hold.iloc[row] = share_hold.iloc[row] * change_ratio
                weight_real.iloc[row] = share_hold.iloc[row] / share_hold.iloc[row].sum()
                fee_detail.loc[date, "current_fee"] = fee
        
        try:
            date_lead1 = date_list[i+1]
        except:
            date_lead1 = str(weight_real.index[-1]).split()[0]
        
        # 对一个周期内的数据进行日频计算
        tmp_fee_daily = 0
        for j in range(1, len(weight_real.loc[date:date_lead1])):
            row_j = row + j
            # 用于判断是否是周频的起始日
            if weight_real.iloc[row_j-1, -1] == 1:
                share_hold.iloc[row_j] = share_hold.iloc[row_j - 1]
                weight_real.iloc[row_j] = weight_real.iloc[row_j - 1]
            else:
                share_hold.iloc[row_j, :-1] = share_hold.iloc[row_j - 1, :-1] * (close_rate.iloc[row_j] + 1)
                share_hold.iloc[row_j, -1] = share_hold.iloc[row_j - 1, -1]
                weight_real.iloc[row_j, :-1] = (share_hold.iloc[row_j, :-1] / share_hold.iloc[row_j, :-1].sum()) * (1 - weight_real.iloc[row_j-1, -1])
                weight_real.iloc[row_j, -1] = 1 - weight_real.iloc[row_j, :-1].sum()
                max_lag = np.max(share_hold.iloc[row_j -param["days_back"]:row_j].sum(axis = 1))
                if 1 - share_hold.iloc[row_j].sum() / max_lag > param["threshold"]:
                    print(f"{weight_real.index[row_j]} is under threshold, short position is needed")
                    share_hold.iloc[row_j, -1] = share_hold.iloc[row_j, :-1].sum()
                    share_hold.iloc[row_j, :-1] = 0
                    weight_real.iloc[row_j, :-1], weight_real.iloc[row_j, -1] = 0, 1
                    
            # 计算每日的费率
            tmp_fee_daily += share_hold.iloc[row_j, :-1].sum() * fee_man / 252
            fee_daily.iloc[row_j, 0] = share_hold.iloc[row_j, :-1].sum() * fee_man / 252
            
            # 计算每日费后净值
            net_position_after_man.iloc[row_j, 0] = share_hold.iloc[row_j].sum() - tmp_fee_daily
        value_lag1 = share_hold.iloc[row_j].sum()
        print(f"backtest from {date} to {date_lead1} has been done")

    net_position_after_man.iloc[0, 0] = param["init_value"]
    net_position = share_hold.sum(axis = 1)
    return_rate = net_position.pct_change()
    fee_detail.iloc[:, 1] = fee_detail.iloc[:, 0].cumsum()
    backtest_result = {"net_value": net_position, 
                       "return_rate": return_rate, 
                       "real_position": weight_real, 
                       "share_hold": share_hold,
                       "accrued_fee": fee_detail,
                       "net_value_after_management_fee": net_position_after_man}
    # 保存回测结果
    with pd.ExcelWriter(param["result_savepath"]) as writer:
        for k, v in backtest_result.items():
            v.to_excel(writer, k)

    plt.plot(net_position_after_man)
    return backtest_result
if __name__ == "__main__":
    param = init_param()
    backtest_result = backtest(param)