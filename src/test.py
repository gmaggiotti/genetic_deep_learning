#import pytest
from GDNN_one_layer import gdnn

gdnn = gdnn(1)

dbpath = "/../dataset/data-500.csv"
X, Y = gdnn.read_dataset(dbpath, 500)
result = gdnn.process(X, Y)

print(result)
