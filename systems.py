import numpy as np

def scaled_system(x_t, u_t, T, U, original_system, th, Ts):
    # Transform back to original variables
    x = np.dot(np.linalg.inv(T), x_t)
    u = np.dot(1/U, u_t) if U.shape == (1,) else np.dot(np.linalg.inv(U), u_t)

    # Compute the next state in original coordinates
    x_n = original_system(x, u, th, Ts)
    
    # Transform the next state back to scaled variables
    x_tn = np.dot(T, x_n[0])
    return x_tn


def simulate_scaled_system(x_t0, u_t_seq, original_system, th, Ts, T=None, U=None):
    N  = u_t_seq.shape[0]
    nu = u_t_seq.shape[1]
    nx = x_t0.shape[0]

    T = np.eye(nx) if T is None else T
    U = np.eye(nu) if U is None else U

    # initialize trajectory vector
    trajectory = np.zeros((N, nx))
    trajectory[0,:] = x_t0

    # initialize input vector
    u_traject = U[0]*u_t_seq if nu==1 else np.dot(U,u_t_seq)

    # generate trajectory
    for idx in range(N-1):
        trajectory[idx+1] = scaled_system(trajectory[idx], u_traject[idx], T, U, original_system, th, Ts)
    
    return trajectory, u_traject


def silverbox_sin(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 + b2*x2 + b3*u +b4*np.sin(x1))

    return np.c_[x1_n, x2_n]


def silverbox_cos(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 + b2*x2 + b3*u +b4*np.cos(x1))

    return np.c_[x1_n, x2_n]


def silverbox_sinx2(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 + b2*x2 + b3*u +b4*np.sin(x2))

    return np.c_[x1_n, x2_n]


def silverbox_sin2x(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 + b2*x2 + b3*u +b4*np.sin(2*x1))

    return np.c_[x1_n, x2_n]


def silverbox_square(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 + b2*x2 + b3*u +b4*x1**2)

    return np.c_[x1_n, x2_n]


def silverbox_sin5x(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 +b2*np.sin(5*x1) +b3*x2 + b4*u)

    return np.c_[x1_n, x2_n]


def silverbox_cube(x,u,th,Ts):
    # coefficients
    [b1, b2, b3, b4] = th
    # current state
    x1, x2 = x
    # state progression
    x1_n = np.array([x1 + Ts*x2])
    x2_n = x2 + Ts*(b1*x1 + b2*x2 + b3*u +b4*x1**3)

    return np.c_[x1_n, x2_n]