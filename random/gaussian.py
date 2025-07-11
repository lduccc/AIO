import numpy as np

def gaussian(nums, mean = 0, standard_deviation = 1):
    first_term = 1 / (standard_deviation * np.sqrt(np.pi * 2))
    second_term = np.power(np.e, (-(nums - mean) ** 2) / (2 * standard_deviation ** 2))

    return np.round(first_term * second_term, 3)

nums = np.array(list(map(float, input().split())))
mean = int(input("Enter mean: "))
std = int(input("Enter standard deviation: "))
print(gaussian(nums, mean, std))