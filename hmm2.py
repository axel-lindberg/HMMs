def viterbi(A, B, pi, obs):
    N = len(A)
    T = len(obs)
    delta = [[0.0] * N for _ in range(T)]
    psi = [[0.0] * N for _ in range(T)]

    #Eq. 2.8
    for i in range(N):
        delta[0][i] = pi[i] * B[i][obs[0]]
    
    for t in range(1, T):
        for i in range(N):
            values = []
            for j in range(N):
                values.append(delta[t - 1][j] * A[j][i])
                
            delta[t][i] = (B[i][obs[t]] * max(values))         # eq. 2.9
            psi[t][i] = values.index(max(values))              # eq. 2.10
       
    x_t = delta[T - 1].index(max(delta[T - 1]))
    path = [x_t]
    for t in range(T - 1, 0, -1):
        x_t = psi[t][x_t]
        path.append(x_t)
    path.reverse()
    
    return path