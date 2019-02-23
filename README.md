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

from GDNN_one_layer import gdnn
gdnn = gdnn(1)

dbpath = "/../dataset/data-500.csv"
X, Y = gdnn.read_dataset(dbpath, 500)
result = gdnn.process(X, Y)

print(result)

```

returns

```python

['1200,0.120717339782,0.857222294615', '1800,0.11497096409,0.846954684894', '2400,0.101505972685,0.841761424897', '3000,0.10017558906,0.843147897998', '3600,0.0996719998151,0.843389434766', '4200,0.0993791016308,0.843680275864', '4800,0.0991470908876,0.843948313688', '5400,0.0989582101751,0.84413401096', '6000,0.0988006379775,0.844216436793']

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
