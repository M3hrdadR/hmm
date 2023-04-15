import numpy as np

N = {}
M = {}
Pi = {}
A = {}
B = {}

def setN(n, idx):
    global N
    N[idx] = n



def setM(m, idx):
    global M
    M[idx] = m


def setA(a, idx):
    global A
    A[idx] = np.array(a)


def setB(b, idx):
    global B
    B[idx] = np.array(b)


def setPi(pi, idx):
    global Pi
    Pi[idx] = np.array(pi)
