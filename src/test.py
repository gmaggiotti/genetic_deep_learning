import pytest
from GDNN_one_layer import gdnn
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


dbpath = "/../dataset/data-500.csv"
X, Y = gdnn.read_dataset(dbpath, 500)
print(X, Y)
result = gdnn.process(X, Y)

print(result)