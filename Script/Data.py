import pandas as pandas
import torch


class csvToData():
    def __init__(self, name):
        self.name = name

    def change(self):
        csv = pandas.read_csv(self.name)
        tensors = torch.tensor(csv.values)

        return tensors
