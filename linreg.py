# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np

from pyspark import SparkContext
from scipy.stats.vonmises_cython import numpy
from numpy import int


if __name__ == "__main__":
    if len(sys.argv) !=2:
        print >> sys.stderr, "Usage: linreg <datafile>"
        exit(-1)

    sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
    yxinputFile = sc.textFile(sys.argv[1])

    yxlines = yxinputFile.map(lambda line: line.split(','))
    yxfirstline = yxlines.first()
    yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength
  
    def findA(line):
        "function to calculate the X * X transpose"
        X = line[1:]
  # inserting 1 for B0      
        X.insert(0,1)
        Xmatrix = np.asmatrix(X, float)
        Xmatrixtrans = Xmatrix.transpose()
        return np.dot(Xmatrixtrans,Xmatrix)
    
    
    def findB(line):
        "function to calculate the X * Y"
        Y = float(line[0])
        X = line[1:]
        X.insert(0,1)
        Xmatrix = np.asmatrix(X, float).transpose()
        return np.dot(Xmatrix,Y)
    
  # dummy floating point array for beta to illustrate desired output format
    beta = np.zeros(yxlength, dtype=float)
  # map function returns the value from findA and the summation by reducebyKey 
  # as we get key value pair as output we pick the value   
    A = yxlines.map(lambda line:(1,findA(line))).reduceByKey(lambda a,b:np.add(a,b)).map(lambda line:line[1])
    refA = A.collect()[0]
    refA = np.asmatrix(refA,float)
  
  # map function returns the value from findB and the summation by reduce key
  # as we get key value pair as output we pick the value   
    B = yxlines.map(lambda line:(2,findB(line))).reduceByKey(lambda a,b:np.add(a,b)).map(lambda line:line[1])
    refB = B.collect()[0]
    refB = np.asmatrix(refB,float)

 # computes the product of inverse of A(X * X transpose) and B(X *Y )
    beta = np.dot(numpy.linalg.inv(refA),refB)
    
    betaref = np.asarray(beta)
    
  # print the linear regression coefficients in desired output format
    print "beta: "
    for coeff in betaref:
        print coeff

    sc.stop()
