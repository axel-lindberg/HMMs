def forward_algorithm(A, B, pi, obs):
    N = len(A)
    T = len(obs)
    alpha = [[0.0] * N for _ in range(T)]
 
    #Eq. 2.5
    first = []
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs[0]]
    
    #Eq. 2.6
    for t in range(1, T):
        for i in range(N):
            sum_N = 0.0
            for j in range(N):
                sum_N += alpha[t - 1][j] * A[j][i]
            alpha[t][i] = (B[i][obs[t]] * sum_N)
        
    return alpha