# Genetic Deep Learning
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/010b7619b1444d20997b281e889c562d)](https://app.codacy.com/app/patelotech/genetic_deep_learning?utm_source=github.com&utm_medium=referral&utm_content=patelotech/genetic_deep_learning&utm_campaign=Badge_Grade_Dashboard)
[![build](https://travis-ci.com/patelotech/genetic_deep_learning.svg?branch=master)](https://travis-ci.org/patelotech/genetic_deep_learning)
[![Coverage Status](https://coveralls.io/repos/github/patelotech/genetic_deep_learning/badge.svg?branch=master)](https://coveralls.io/github/patelotech/genetic_deep_learning?branch=master)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/gmaggiotti/genetic_deep_learning/blob/master/LICENSE.md)
[![pv](https://img.shields.io/badge/python-2.7-blue.svg)]()

This project aims to use genetic algorithms to boost the learning of DNN.  Building and training a family  of NN with same structure and hyperparameters from scratch but starting from different random weights.   After a few epochs of training, the networks that perform better are chosen and crossover their weights in order to mating and produce the next generation. 

Main problems to solve with NN:

-   Architecture optimization:finding optimal layers and number of nodes in each layer of the network required to capture features from given data.
-   Hyperparameter optimization: refers to choosing values of hyperparameters like - learning rate, optimization algorithm, dropout rate, batch size, etc. 
-   Weight optimization: find the right values for each neuron within each weight in order to solve the general equation with a minimum error.

This project is focused on solving weight optimization, using Genetic Algorithms combined with Gradient Descent and implement a method to make the process faster.

## Intuition of how gradients of the new generations improves the chosen slope when mating the fittest

 The ”Survival of the Fittest” scheme, proposed by Charles Darwin in his Theory of Natural Selection, is used.  The mating process takes place after every ’n’ epoch, in this example n=600. After the first generation of parents are trained for ’n’ iterations, their dominance is calculated based on their ability to reduce loss.  Networks that achieve less loss are selected to create the new generation.
![](img/image2.png)

Comparison of the loss of the GDNN, choosing the best NN within each generation (where each generations occurs within 600 epochs),  vs the loss of DNN.

![](img/error2.png)
![](img/acc2.png)

## Set-up

` git clone https://github.com/gmaggiotti/genetic_deep_learning `

` pip install -r requirements.txt `

## Example Usage

```python
from NN1 import NN1
from sklearn.model_selection import train_test_split
from nn_utils import run_GDNN_model, read_dataset

X, Y = read_dataset(180, 500)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)

epochs = 600
population_size = 10
generations = 10

dataset = train_x, train_y, test_x, test_y
run_GDNN_model(NN1, epochs, population_size, generations, dataset)

```
results:

```python

600,0.173792495024,(0.8137978069627971, 0.4219032949450148)
1200,0.16248481196,(0.8160476844138655, 0.4193646765300602)
1800,0.15799799467,(0.8141032269154518, 0.4238596169932088)
2400,0.15666446084,(0.8142474709241911, 0.4238323175273134)
3000,0.155885024449,(0.8143746488046282, 0.4237553716584507)
3600,0.155346503634,(0.8144767916131259, 0.4236917660134981)
4200,0.154944886609,(0.8145587342577347, 0.4236415894437359)
4800,0.154630396953,(0.814625525590064, 0.4236015808214171)
5400,0.154375301644,(0.8146809956649543, 0.4235690385377556)
6000,0.154162711144,(0.8147279092134139, 0.4235420163038033)


```

## Linting

-   **Style:** PEP8
[PEP8](https://www.python.org/dev/peps/pep-0008/ "Pep 8")

### Versioning

-   pylint 2.2.2
-   astroid 2.1.0
-   autopep8 1.4.3 (pycodestyle: 2.4.0)

### Linting scripts

-   Error check: `pylint src`
-   Error fix:  `autopep8 --in-place --aggressive --aggressive src`

Copyright (c) 2018 Gabriel A. Maggiotti
