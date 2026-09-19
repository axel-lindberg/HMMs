def forward_algorithm(A, B, pi, obs):
    alpha = []
    N = len(A)
    T = len(obs)
    
    #Eq. 2.5
    first = []
    for i in range(N):
        first.append(pi[i] * B[i][obs[0]])
    alpha.append(first)
    
    #Eq. 2.6
    for t in range(1, T):
        alpha_t = []
        for i in range(N):
            sum_N = 0.0
            for j in range(N):
                sum_N += alpha[t - 1][j] * A[j][i]
            alpha_t.append(B[i][obs[t]] * sum_N)
        alpha.append(alpha_t)
        
    return alpha