import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

feature = "local_x"
# 加载训练损失
train_loss = np.load("./result/" + feature + "/train_loss.npy")

plt.plot(train_loss, label='trian_loss')
plt.show()

# 加载预测结果和真实值
pred = np.load("./result/" + feature + "/pred.npy")
true = np.load("./result/" + feature + "/true.npy")

plt.plot(pred[:100], label='pred')
plt.plot(true[:100], label='true')
plt.legend()
plt.show()
