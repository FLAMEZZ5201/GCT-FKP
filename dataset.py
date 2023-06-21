from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

data_path = "train_gsdata_rad.csv"

# Complete the dataset class
class MyDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv(data_path).values  # DataFrame type, values to numpy

    def __getitem__(self, index):
        """
        Gets a piece of data at the corresponding location of the index
        :param index:
        :return:
        """
        return MyDataset.to_tensor(self.data[index])

    def __len__(self):
        """
        Get dataset size 
        :return:
        """
        return len(self.data)

    @staticmethod
    def to_tensor(data):
        """
        ndarray to tensor
        :param data:
        :return:
        """
        return torch.from_numpy(data)

if __name__ == "__main__":
    data = MyDataset() 
    print(data[1]) 
    print(len(data))
