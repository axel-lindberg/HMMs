def forward_algorithm(A, B, pi, obs):
    N = len(A)
    T = len(obs)
    alpha = [[0.0] * N for _ in range(T)]
    c = [0.0] * T
 
    #Compute alpha_0(i)
    first = []
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs[0]]
        c[0] += alpha[0][i]
    
    #Scale the alpha_0(i)
    c[0] = 1.0 / c[0]
    for i in range(N):
        alpha[0][i] *= c[0]
        
    #Compute alpha_t(i)
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                alpha[t][i] += alpha[t - 1][j] * A[j][i]
            alpha[t][i] *= B[i][obs[t]]
            c[t] += alpha[t][i]
            
        #Scale alpha_t(i)
        c[t] = 1.0 / c[t]
        for i in range(N):
            alpha[t][i] *= c[t]
        
    return alpha, c