import sys
import math
import random
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

def read_obs_file(path):
    tokens = open(path).read().split()
    T = int(tokens[0])
    return [int(x) for x in tokens[1:1 + T]]

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
    
    A, B, pi, _ = baum_welch(A, B, pi[0], obs)
    
    print(len(A), len(A[0]), *(v for row in A for v in row))
    print(len(B), len(B[0]), *(v for row in B for v in row))
    
def exp_1():
    A0 = [[0.54, 0.26, 0.20], [0.19, 0.53, 0.28], [0.22, 0.18, 0.60]]
    B0 = [[0.50, 0.20, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.45]]
    PI0 = [0.3, 0.2, 0.5]

    for path in ["hmm_c_N1000.in", "hmm_c_N10000.in"]:
        obs = read_obs_file(path)
        A, B, pi, num_iters = baum_welch(A0, B0, PI0, obs)
        _, c = forward_algorithm(A, B, pi, obs)
        print(path)
        print("log-likelihood per observation =", log_likelihood(c) / len(obs))
        print("A =", A)
        print("B =", B)
        print("pi =", pi)
        print("number of iterations =", num_iters)
        
def exp_2():
    obs = read_obs_file("hmm_c_N10000.in")

    uniform = ([[1/3]*3 for _ in range(3)], [[1/4]*4 for _ in range(3)], [1/3, 1/3, 1/3])
    zeros = ([[1,0,0],[0,1,0],[0,0,1]], [[0.25]*4 for _ in range(3)], [1,0,0])
    close = ([[0.72,0.04,0.24],[0.12,0.77,0.11],[0.22,0.28,0.50]], [[0.68,0.21,0.09,0.02],[0.11,0.42,0.28,0.19],[0.02,0.11,0.22,0.65]], [1,0,0])

    for name, (A0, B0, pi0) in [("uniform", uniform), ("zeros", zeros), ("close", close)]:
        A, B, pi, _ = baum_welch(A0, B0, pi0, obs)
        _, c = forward_algorithm(A, B, pi, obs)
        print(name)
        print("log-likelihood per observation =", log_likelihood(c) / len(obs))
        print("A =", A)
        print("B =", B)
            
def exp_3():
    obs_train = read_obs_file("hmm_c_N10000.in")
    obs_eval = read_obs_file("hmm_c_N1000.in")
    M = 4

    for N in [2, 3, 4]:
        random.seed(42)
        A0 = [[random.random() for _ in range(N)] for _ in range(N)]
        A0 = [[v / sum(row) for v in row] for row in A0]
        B0 = [[random.random() for _ in range(M)] for _ in range(N)]
        B0 = [[v / sum(row) for v in row] for row in B0]
        pi0 = [random.random() for _ in range(N)]
        pi0 = [v / sum(pi0) for v in pi0]

        A, B, pi, _ = baum_welch(A0, B0, pi0, obs_train)
        _, c_train = forward_algorithm(A, B, pi, obs_train)
        _, c_eval = forward_algorithm(A, B, pi, obs_eval)

        print(N, "train log-likelihood per observation =", log_likelihood(c_train) / len(obs_train),
                 "eval log-likelihood per observation =", log_likelihood(c_eval) / len(obs_eval))
        print("A =", A)
        print("B =", B)

def main():
    # hmm0()
    # hmm1()
    # hmm2()
    # hmm3()
    
    #exp_1()
    exp_2()
    # exp_3()
    
if __name__ == "__main__":
    main()