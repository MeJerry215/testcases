from matplotlib import gridspec
import pylab
from sklearn.datasets import make_classification
import numpy as np
import pickle
import os
import gzip
import random
import pdb
from tqdm import tqdm

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

n_samples = 100
X, Y = make_classification(n_samples=n_samples, n_features=2,
                            n_redundant=0, n_informative=2, flip_y=0)
Y = Y * 2 - 1
X = X.astype(np.float32)
Y = Y.astype(np.int32)
train_x, test_x = np.split(X, [n_samples * 8 // 10])
train_y, test_y = np.split(Y, [n_samples * 8 // 10])
print("Features:\n",train_x[0:4])
print("Labels:\n",train_y[0:4])


def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = pylab.subplots(1, 1)
    #pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')

    colors = ['r' if l>0 else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha = 0.5)
    fig.savefig('sample_plot.png')
    # fig.save


plot_dataset("train data", train_x, train_y)
pos_examples = np.array([ [t[0], t[1], 1] for i,t in enumerate(train_x) if train_y[i]>0])
neg_examples = np.array([ [t[0], t[1], 1] for i,t in enumerate(train_x) if train_y[i]<0])

pdb.set_trace()

def train(positive_examples, negative_examples, num_iters=100):
    num_dims = pos_examples.shape[1]
    weights = np.zeros((num_dims,1))
    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]
    report_frequency = 10
    for i in tqdm(range(num_iters)):
        # Pick one positive and one negative example
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)
        z = np.dot(pos, weights)
        if z < 0: # positive example was classified as negative
            weights = weights + pos.reshape(weights.shape)

        z  = np.dot(neg, weights)
        if z >= 0: # negative example was classified as positive
            weights = weights - neg.reshape(weights.shape)

        # Periodically, print out the current accuracy on all examples
        if i % report_frequency == 0:
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            print("Iteration={}, pos correct={}, neg correct={}".format(i,pos_correct,neg_correct))

train(pos_examples, neg_examples, num_iters=200)