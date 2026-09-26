from hmm1 import forward_algorithm
from helper import log_likelihood

def backward_algorithm(A, B, obs, N, T, c):
    beta = [[0.0] * N for _ in range(T)]
    
    for i in range(N):
        beta[T - 1][i] = c[T - 1]
        
    for t in range(T - 2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]
            beta[t][i] *= c[t]
            
    return beta

def compute_gammas(alpha, beta, A, B, obs, N, T):
    gamma = [[0.0] * N for _ in range(T)]
    di_gamma = [[[0.0] * N for _ in range(N)] for _ in range(T)]
    
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                di_gamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]       #eq 2.14    
                gamma[t][i] += di_gamma[t][i][j]                                                    #eq 2.15
    
    #eq 2.16   
    for i in range(N):  
        gamma[T - 1][i] = alpha[T - 1][i]
        
    return gamma, di_gamma

def reestimate(gamma, di_gamma, obs, N, M, T):
    #re-estimate pi
    pi = [gamma[0][i] for i in range(N)]
    
    #re-estimate A
    A = [[0.0] * N for _ in range(N)]
    for i in range(N):
        denom = 0.0
        for t in range(T - 1):
            denom += gamma[t][i]
            
        for j in range(N):
            numer = 0.0
            for t in range(T - 1):
                numer += di_gamma[t][i][j]
        
            A[i][j] = numer / denom
        
    #re-estimate B
    B = [[0.0] * M for _ in range(N)]
    for i in range(N):
        denom = 0.0
        for t in range(T):
            denom += gamma[t][i]
            
        for j in range(M):
            numer = 0.0
            for t in range(T):
                if obs[t] == j:
                    numer += gamma[t][i]
        
            B[i][j] = numer / denom
            
    return A, B, pi

def baum_welch(A, B, pi, obs, max_iters=100):
    N = len(A)
    M = len(B[0])
    T = len(obs)
    
    old_log_prob = float('-inf')
    
    for _ in range(max_iters):
        alpha, c = forward_algorithm(A, B, pi, obs)
        beta = backward_algorithm(A, B, obs, N, T, c)
        gamma, di_gamma = compute_gammas(alpha, beta, A, B, obs, N, T)
        A, B, pi = reestimate(gamma, di_gamma, obs, N, M, T)
        
        log_prob = log_likelihood(c)
        if log_prob > old_log_prob:
            old_log_prob = log_prob
        else:
            break
    
    return A, B, pi
    