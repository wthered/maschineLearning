import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


class SupportVectorMachine:
	w = 0
	b = 0

	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b'}
		self.data = None
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1, 1, 1)

	# train
	def fit(self, data):
		self.data = data
		pass

	def predict(self, features):
		# sign( x.w+b )
		return np.sign(np.dot(np.array(features), self.w) + self.b)


if __name__ == '__main__':
	data_dict = {
		-1: np.array([
			[1, 7],
			[2, 8],
			[3, 8],
		]),
		1: np.array([
			[5, 1],
			[6, -1],
			[7, 3]
		])
	}
	svm = SupportVectorMachine()
	svm.fit(data=data_dict)
	print(svm.predict([1, -1]))
