from scipy.sparse import random
from itertools import zip_longest
import argparse
import numpy
import csv

class Args:
    pass

parser = argparse.ArgumentParser(description='Generate random sparse matrix.')
parser.add_argument('m', type=int, help='number of rows')
parser.add_argument('n', type=int, help='number of columns')
parser.add_argument('-d', '--density', type=float, help='matrix density (default is 0.1)', default=0.1)
args = Args()
parser.parse_args(namespace=args)

m, n, density = args.m, args.n, args.density
matrix = random(m, n, format='csr', density=density)
values = matrix.data
rowPtr = matrix.indptr
colInd = matrix.indices

print(matrix.todense())
print(values)
print(rowPtr)
print(colInd)
# Write to CSV file
filename = '{}x{}.csv'.format(args.m, args.n)
with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['row', 'col', 'value'])
    for (row, col, val) in zip_longest(rowPtr, colInd, values):
        if row == None:
            row = '-'
        writer.writerow([row, col, val])