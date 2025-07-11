import numpy as np
from collections import defaultdict
np.random.seed(69420)

def roll_n_dices(n):
    '''n dices thrown simulation'''
    dices = np.random.randint(1, 7, n)
    return dices

def probability(dices):
    '''return the proportion of each side given the dices'''
    occurences = np.zeros(6)
    for die in dices:
        occurences[die - 1] += 1
    return occurences / len(dices) * 100

n = int(input("How many dices you want to roll: "))
print(probability(roll_n_dices(n)))

