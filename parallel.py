import multiprocessing
import numpy as np

from itertools import product

def pi_mc(seed, dummy):
    np.random.seed(seed)
    num_trials = 1000
    counter = 0
    for j in range(num_trials):
        x_val = np.random.random()
        y_val = np.random.random()

        radius = x_val**2 + y_val**2

        if radius < 1:
            counter += 1

    return [4*counter/num_trials]*2

with multiprocessing.Pool() as pool:
    result = pool.starmap(pi_mc, list(product(range(12), range(12))))

real_result = sum(result, [])
print(real_result)

a = [1,2]
b = ["hi", "bye"]
c = [-1, -2, -3]
list(zip(*product(a,b)))

list(zip(*zip(*product(a,b)), c))

unzip(list(product(a,b)))
