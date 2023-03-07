from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

data_path = "train_gsdata_rad.csv"

# 完成数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv(data_path).values  # DataFrame类型，通过values转换成numpy类型

    def __getitem__(self, index):
        """
        必须实现，作用是:获取索引对应位置的一条数据
        :param index:
        :return:
        """
        return MyDataset.to_tensor(self.data[index])

    def __len__(self):
        """
        必须实现，作用是得到数据集的大小
        :return:
        """
        return len(self.data)

    @staticmethod
    def to_tensor(data):
        """
        将ndarray转换成tensor
        :param data:
        :return:
        """
        return torch.from_numpy(data)

if __name__ == "__main__":
    data = MyDataset() # 实例化对象
    print(data[1]) # 取第1条数据
    print(len(data)) # 获取长度
