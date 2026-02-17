#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
1. 有没有重复 timestamp
2. 5min 是否连续（有没有缺 bar）
3. volume / num_trades 是否有 0 或极端异常
4. 简单统计：mean / std / quantile
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import time

def main():
    start_month = '2025-01'
    end_month = '2025-12'
    symbol = 'BTCUSDT'
    interval = '5m'
    FREQ = '5min'
    data_path = f'./data/{symbol}/{interval}'
    data_files = os.listdir(data_path)
    data_files.sort()   
    data_files = [data_file for data_file in data_files if data_file.startswith('k')]
    print(f'Found {len(data_files)} data files')
    print(f'Data files: {data_files}')
    counter = 0
    for data_file in data_files:
        counter += 1
        print(counter)
        data_file_path = os.path.join(data_path, data_file)
        data = pd.read_csv(data_file_path)
        # 1. 有没有重复 timestamp
        data['open_time'] = pd.to_datetime(data['open_time'], utc=True)
        num_duplicates = data['open_time'].duplicated().sum()
        print(f"Duplicate timestamps: {num_duplicates}")
        if num_duplicates > 0:
            print(f"WARNING: duplicated timestamps found in {data_file}")
            continue
        
        # 2. 5min 是否连续（有没有缺 bar）
        data = data.sort_values('open_time').reset_index(drop=True)
        time_diff = data['open_time'].diff().dropna()
        expected = pd.Timedelta(FREQ)
        num_gaps = (time_diff != expected).sum()
        print(f"Non-5min gaps: {num_gaps}")

        # 3. volume / num_trades 是否有 0 或极端异常
        nan_counts = data.isna().sum()
        print(nan_counts[nan_counts > 0])

        # 4. 简单统计：mean / std / quantile
        print(data.describe())


if __name__ == "__main__":
    main()  