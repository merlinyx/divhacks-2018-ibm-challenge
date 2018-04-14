# DivHacks 2018 IBM Challenge. 
# Yuxuan Mei. 
# 2018.4.14

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_fin = '/Users/Bluefish_/divhacks/MNIST_test.csv'
test_fout = '/Users/Bluefish_/divhacks/divhacks-2018-ibm-challenge/test_label.csv'
train_fin = '/Users/Bluefish_/divhacks/MNIST_train.csv'
train_fout = '/Users/Bluefish_/divhacks/divhacks-2018-ibm-challenge/train_label.csv'
THRESHOLD = 0.5

def calc_easy(fin, fout):
	''' calculate easy/hard labels with THRESHOLD '''
	file = open(fout, 'w', newline='')
	filewriter = csv.writer(file, delimiter=',')
	with open(fin, 'r', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		first_line = True
		for row in reader:
			if first_line:
				row.append('PC')
				row.append('EASYHARD')
				filewriter.writerow(row)
				first_line = False
			else:
				count = 0
				for i in range(2, len(row)):
					count += int(row[i])
				PC = count / 21.0
				row.append(PC)
				# 0 - easy, 1 - hard
				row.append(0 if PC > THRESHOLD else 1)
				filewriter.writerow(row)
	file.close()

def count_easy(fin):
	''' count easy/hard labels for each digit and plot. '''
	counts = []
	counts_hard = []
	total = [0] * 10
	hard = [0] * 10

	with open(fin, 'r', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		first_line = True
		for row in reader:
			if first_line:
				first_line = False
			else:
				counts.append(int(row[1]))
				total[int(row[1])] += 1
				if int(row[-1]):
					# 0 - easy, 1 - hard
					counts_hard.append(int(row[1]))
					hard[int(row[1])] += 1
	
	frac = np.array([hard[i] / total[i] for i in range(10)])
	print(np.argsort(frac))
	# print('The hardest digit to predict is {}. '.format(np.argmax(frac)))
	# print('The easiest digit to predict is {}. '.format(np.argmin(frac)))

	return counts, counts_hard

def plot_count(fname, counts, counts_hard):
	''' plot the counts for each digits '''
	plt.figure()
	plt.grid(True)
	plt.style.use('seaborn-deep')
	x = np.array(counts)
	y = np.array(counts_hard)
	plt.hist([x, y], 10, histtype='bar',label=['Count_Total', 'Count_Hard'])
	plt.legend(loc='upper right')
	plt.savefig(fname)
	print('{} is saved. '.format(fname))

def cross_val_knn():
	''' knn classifier and cross validation'''
	dataset = pd.read_csv(train_fout, sep=',')
	X, y = dataset.iloc[:,2:-2], dataset.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

	# k-list
	kk = [1, 3, 5, 7, 9]

	# empty list that will hold cv scores
	cv_scores = []

	# perform 10-fold cross validation
	for k in kk:
	    knn = KNeighborsClassifier(n_neighbors=k)
	    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
	    cv_scores.append(scores.mean())

	# changing to error
	err = [1 - x for x in cv_scores]

	# determining best k
	optimal_k = kk[err.index(min(err))]
	print('The optimal number of neighbors is {}. '.format(optimal_k))

	# plot error vs k
	plt.figure(figsize=(8, 5))
	plt.plot(kk, err)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('Misclassification Error')
	plt.savefig('cross_val.png')

def binary_classification():
	knn = KNeighborsClassifier(n_neighbors=1)
	
	dataset_train = pd.read_csv(train_fout, sep=',')
	X_train, y_train = dataset_train.iloc[:,2:-2], dataset_train.iloc[:, -1]
	dataset_test = pd.read_csv(test_fout, sep=',')
	X_test, y_test = dataset_test.iloc[:,2:-2], dataset_test.iloc[:, -1]
	
	knn.fit(X_train, y_train)
	train_pred = knn.predict(X_train)
	test_pred = knn.predict(X_test)
	print('Training accuracy: {}'.format(accuracy_score(y_train, train_pred)))
	print('Test accuracy: {}'.format(accuracy_score(y_test, test_pred)))

if __name__ == '__main__':
	''' calculate easy/hard labels with THRESHOLD '''
	# calc_easy(test_fin, test_fout)
	# calc_easy(train_fin, train_fout)

	''' plot the counts for each digits '''
	# test_c, test_ch = count_easy(test_fout)
	# plot_count('test_hist.png', test_c, test_ch)

	# train_c, train_ch = count_easy(train_fout)
	# plot_count('train_hist.png', train_c, train_ch)

	# test_c.extend(train_c)
	# test_ch.extend(train_ch)
	# plot_count('total_hist.png', test_c, test_ch)

	''' cross validation to find the k for knn '''
	# cross_val_knn()

	''' use the knn to predict and report accuracies '''
	# binary_classification()
