import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import style

style.use('fivethirtyeight')


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ylist = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ylist.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xlist = [i for i in range(len(ylist))]
	return np.array(xlist, dtype=np.float64), np.array(ylist, dtype=np.float64)


def best_fit_slope_intercept(x, y):
	slope = ((mean(x) * mean(y)) - mean(x * y)) / (mean(x) ** 2 - mean(x ** 2))
	error = mean(y) - slope * mean(x)
	return round(slope, 3), round(error, 3)


def squared_error(ys_original, ys_line):
	return sum((ys_line - ys_original) ** 2)


xs, ys = create_dataset(40, 40, 2, True)
a, b = best_fit_slope_intercept(xs, ys)
regression_line = [(a * x + b) for x in xs]
print(a, b)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
