import sys
import math
from helper import log_likelihood
from hmm0 import next_observation_distribution
from hmm1 import forward_algorithm
from hmm2 import viterbi
from hmm3 import baum_welch

def read_matrix(lines):
    tokens = lines.split()
    index = 0
    
    rows, cols = int(tokens[index]), int(tokens[index + 1])
    index += 2
    
    M = []
    for r in range(rows):
        row = []

        for c in range(cols):
            row.append(float(tokens[index]))
            index += 1

        M.append(row)
    
    return M

def read_sequence(line):
    tokens = line.split()
    index = 0
    
    T = int(tokens[index])
    index += 1
    
    seq = []
    
    for t in range(0, T):
        seq.append(int(tokens[index]))
        index += 1

    return seq

def hmm0():
    lines = sys.stdin.read().splitlines()
    A = read_matrix(lines[0]) #transition matrix
    B = read_matrix(lines[1]) #emission matrix
    pi = read_matrix(lines[2]) #initial state

    p = next_observation_distribution(A, B, pi)
    
    print(1, len(p), *p)
    
def hmm1():
    lines = sys.stdin.read().splitlines()
    A = read_matrix(lines[0])
    B = read_matrix(lines[1])
    pi = read_matrix(lines[2])
    obs = read_sequence(lines[3]) #sequence of emissions
    
    alpha, c = forward_algorithm(A, B, pi[0], obs)
    print(math.exp(log_likelihood(c)))
    
def hmm2():
    lines = sys.stdin.read().splitlines()
    A = read_matrix(lines[0])
    B = read_matrix(lines[1])
    pi = read_matrix(lines[2])
    obs = read_sequence(lines[3])
    
    path = viterbi(A, B, pi[0], obs)
    
    print(*path)
    
def hmm3():
    lines = sys.stdin.read().splitlines()
    A = read_matrix(lines[0])
    B = read_matrix(lines[1])
    pi = read_matrix(lines[2])
    obs = read_sequence(lines[3])
    
    A, B, pi = baum_welch(A, B, pi[0], obs)
    
    print(len(A), len(A[0]), *(v for row in A for v in row))
    print(len(B), len(B[0]), *(v for row in B for v in row))

def main():
    # hmm0()
    # hmm1()
    # hmm2()
    hmm3()
    
if __name__ == "__main__":
    main()