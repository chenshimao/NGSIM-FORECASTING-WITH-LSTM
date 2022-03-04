import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 数据处理
root_path = r"./ngsim_data/部分数据"
all_file_list = os.listdir(root_path)
all_data_frame = pd.DataFrame()
for file in all_file_list:
    single_data_frame = pd.read_csv(os.path.join(root_path, file))
    if file == all_file_list[0]:
        all_data_frame = single_data_frame
    else:
        all_data_frame = pd.concat([all_data_frame, single_data_frame], axis=0)

all_data_frame = all_data_frame.dropna(axis=0, how="any")
all_data_frame.to_csv("./data/data.csv")
