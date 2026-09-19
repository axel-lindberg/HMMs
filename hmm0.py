from helper import vec_mat_mult

def next_observation_distribution(A, B, pi):
    return vec_mat_mult(vec_mat_mult(pi[0], A), B)