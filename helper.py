def vec_mat_mult(v, M):
    cols = len(M[0])
    result = []
    
    for c in range(cols):
        value = 0.0
        for r in range(len(v)):
            value += v[r] * M[r][c]
        result.append(value)
    return result