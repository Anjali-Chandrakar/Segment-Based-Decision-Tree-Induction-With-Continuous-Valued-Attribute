"""
Created on Wed Mar 14 13:42:50 2018

@author: acer
"""
from csv import reader
import math
from random import seed
from random import randrange
from operator import itemgetter
import sys
from sklearn.metrics import confusion_matrix

# Calculate the Gain index for a split dataset
def info_value(groups, classes,split):
	instances = float(sum([len(g) for g in groups]))
	info = 0.0
	for g in groups:
		size = float(len(g))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in g].count(class_val) / size
			if p!=0:
				score += p * math.log(p,2)
		info += (-score) * (size / instances)
	if split==0:
		gain=0
	else:
		gain = (info_d - info)/split
	return gain

# Select the best split point for a dataset
def get_split(dataset):
	b_index, b_value, b_score, b_groups = 0, 0, -100, None
	#attr_values=list(zip(*dataset))
	for index in range(len(dataset[0])-1):
		dataset=sorted(dataset,key=lambda dataset:dataset[index])	
		class_values = list(set(row[-1] for row in dataset))
		for i in range(len(dataset)-1):
			split=0.0
			groups,split = test_split(index, (dataset[i][index]+dataset[i+1][index])/2, dataset)
			gain = info_value(groups, class_values, split)
			if gain > b_score:
				b_index, b_value, b_score, b_groups = index, (dataset[i][index]+dataset[i+1][index])/2, gain, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
	
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] <= value:
			left.append(row)
		else:
			right.append(row)
	p_l=len(left)/len(dataset)
	p_r=len(right)/len(dataset)
	if p_l==0: 
		split1=0
	else:
		split1 = -(p_l * math.log(p_l,2))
	if p_r==0:
		split2=0
	else:
		split2 = -(p_r * math.log(p_r,2))
	split=split1+split2
	return (left, right),split
	
# Load a CSV file
def load_file(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Split a dataset into k folds
def split_data(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
	

# Calculate accuracy percentage
def cal_accuracy(actual, predicted,Y,Y_pred):
	correct = 0
	for i in range(len(actual)):
		Y.append(actual[i])
		Y_pred.append(predicted[i])
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0,Y,Y_pred
		
# Evaluate an algorithm using a cross validation split
def evaluating_algo(dataset, algorithm, n_folds, *args):
	folds = split_data(dataset, n_folds)
	scores = list()
	Y=[]
	Y_pred=[]
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy,Y,Y_pred = cal_accuracy(actual, predicted,Y,Y_pred)
		scores.append(accuracy)
	print(confusion_matrix(Y, Y_pred))
	print("Precision : ",confusion_matrix(Y, Y_pred)[0,0]/(confusion_matrix(Y, Y_pred)[0,0]+confusion_matrix(Y, Y_pred)[0,1]))
	print("Recall : ",confusion_matrix(Y, Y_pred)[0,0]/(confusion_matrix(Y, Y_pred)[0,0]+confusion_matrix(Y, Y_pred)[1,0]))
	return scores
 
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
	 
# Create a terminal node value
def to_terminal(g):
	outcomes = [row[-1] for row in g]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	left_class=[]
	for i in range(len(left)):
		left_class.append(left[i][-1])
	right_class=[]
	for i in range(len(right)):
		right_class.append(right[i][-1])
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(set(left_class))==1:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(set(right_class))==1:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
		
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)
 
seed(1)
# load and prepare data
filename = 'SPECTF_test.csv'
#filename = 'bupa.csv'
#filename = 'haberman.csv'

dataset = load_file(filename)

classes = list(set(row[-1] for row in dataset))
size_d=float(len(dataset)-1)
score_d=0.0
for class_val in classes:
	p = [row[-1] for row in dataset].count(class_val) / size_d
	score_d += p * math.log(p,2)
info_d=(-score_d)

# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_to_float(dataset, i)

# evaluate algorithm
n_folds = 10
max_depth = 100
min_size = 10
print("##########################  C4.5 Algorithm  ##########################")
print("Dataset : ",filename)
scores = evaluating_algo(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
	
	