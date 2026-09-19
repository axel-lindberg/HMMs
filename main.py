import sys
from hmm0 import next_observation_distribution

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

def format_matrix(M):
    rows = len(M)
    cols = len(M[0])

    values = []

    for row in M:
        for value in row:
            values.append(f"{value:.1f}")

    return f"{rows} {cols} " + " ".join(values)

def format_vector(v):
    cols = len(v)
    
    values = []
    
    for value in v:
        values.append(f"{value:.1f}")
        
    return f"1 {cols} " + " ".join(values)

def main():
    lines = sys.stdin.read().splitlines()
    
    A = read_matrix(lines[0])
    B = read_matrix(lines[1]) 
    pi = read_matrix(lines[2])

    p = next_observation_distribution(A, B, pi)
    
    print(format_vector(p))
    
if __name__ == "__main__":
    main()