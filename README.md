# Genetic Deep Learning
[![build](https://travis-ci.com/patelotech/genetic_deep_learning.svg?branch=master)](https://travis-ci.org/patelotech/genetic_deep_learning)
This project aims to use genetic algorithms to boost the learning of DNN.  Building and training a family  of NN with same structure and hyperparameters from scratch but starting from different random weights.   After a few epochs of training, the networks that perform better are chosen and crossover their weights in order to mating and produce the next generation. 

Main problems to solve with NN:

- Architecture optimization:finding optimal layers and number of nodes in each layer of the network required to capture features from given data.
- Hyperparameter optimization: refers to choosing values of hyperparameters like - learning rate, optimization algorithm, dropout rate, batch size, etc. 

- Weight optimization: find the right values for each neuron within each weight in order to solve the general equation with a minimum error.

This project is focused on solving weight optimization, using Genetic Algorithms combined with Gradient Descent and implement a method to make the process faster.


## Intuition of how gradients of the new generations improves the chosen slope when mating the fittest

 The ”Survival of the Fittest” scheme, proposed by Charles Darwin in his Theory of Natural Selection, is used.  The mating process takes place after every ’n’ epoch, in this example n=600. After the first generation of parents are trained for ’n’ iterations, their dominance is calculated based on their ability to reduce loss.  Networks that achieve less loss are selected to create the new generation.
![](img/image2.png)

Comparison of the loss of the GDNN, choosing the best NN within each generation (where each generations occurs within 600 epochs),  vs the loss of DNN.
![](img/loss.png)

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

