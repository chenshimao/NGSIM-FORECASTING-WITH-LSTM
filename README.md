# NGSIM FORECASTING WITH LSTM

# 一、项目目录结构

![Untitled](NGSIM%20FORE%20c66d7/Untitled.png)

data：用于将数据按照滑动窗口大小划分标准的数据格式

models：存放模型

ngsim_data：存放原始数据

result：保存结果（pred为预测结果，true为真实值，train_loss为训练损失）

utils：工具类

draw.py：画图

main1.py：启动运行，用于预测local_x

main2.py：启动运行，用于预测local_y

main3.py：启动运行，用于预测v_Vel

process.py：用于数据预处理

# 二、使用说明

1. 首先运行process.py文件，该程序会在data目录下生成data.csv文件，该文件保存了所有用于训练的数据
2. 分别运行main1.py main2.py main3.py文件，运行后可在result目录下生成.npy的结果
3. 运行draw.py可以生成所需图表

# 三、main.py运行指令

```bash
python main1.py --seq_len 30 --hidden_size 256 --num_layers 2 --dropout 0.05 --is_bidirectional Flase --lr 0.0001 --epoches 5 --batch_size 16
```

注：各参数含义在main.py中均有解释