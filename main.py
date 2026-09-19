import sys
from hmm0 import next_observation_distribution
from hmm1 import forward_algorithm

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

def format_matrix(M):
    rows = len(M)
    cols = len(M[0])

    values = []

    for row in M:
        for value in row:
            values.append(f"{value:.6f}")

    return f"{rows} {cols} " + " ".join(values)

def format_vector(v):
    cols = len(v)
    
    values = []
    
    for value in v:
        values.append(f"{value:.6f}")
        
    return f"1 {cols} " + " ".join(values)

def main():
    lines = sys.stdin.read().splitlines()
    
    #HMM0
    # A = read_matrix(lines[0]) #transition matrix
    # B = read_matrix(lines[1]) #emission matrix
    # pi = read_matrix(lines[2]) #initial state

    # p = next_observation_distribution(A, B, pi)
    
    # print(format_vector(p))
    
    #HMM1
    A = read_matrix(lines[0]) #transition matrix
    B = read_matrix(lines[1]) #emission matrix
    pi = read_matrix(lines[2]) #initial state
    obs = read_sequence(lines[3]) #sequence of emissions
    
    alpha = forward_algorithm(A, B, pi[0], obs)
    print(sum(alpha[-1]))
    
if __name__ == "__main__":
    main()