import numpy as np
import matplotlib.pyplot as plt
from quiver_with_circ import dz_dot, theta_dot_of_dz, w_dot, theta_dot

lambda_ = 1
r = 1
gamma = 1
k = 1
dt = 0.01
T = 100

alpha_init = 2

W_trajectory = [np.sqrt(r / 2) / alpha_init]
Theta_trajectory = [np.sqrt(r / 2)]

W_trajectory_rep = W_trajectory.copy()
Theta_trajectory_rep = Theta_trajectory.copy()

dalpha = -0.13

alphas = np.arange(alpha_init, -alpha_init, dalpha)
alpha_trajectory = [alphas[0]]

for alpha in alphas[1:]:
    print(f'{alpha:.2f}')
    epsilon = 1e-4
    converge = False
    while not converge:
        W_dot_t = w_dot(W_trajectory[-1], Theta_trajectory[-1], alpha, lambda_, r, k=1, gamma=1)
        Theta_dot_t = theta_dot(W_trajectory[-1], Theta_trajectory[-1], alpha, lambda_, r, gamma=1)

        W_trajectory.append(W_trajectory[-1] + W_dot_t * dt)
        Theta_trajectory.append(Theta_trajectory[-1] + Theta_dot_t * dt)

        if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
            break

    W_trajectory_rep.append(W_trajectory[-1])
    Theta_trajectory_rep.append(Theta_trajectory[-1])
    alpha_trajectory.append(alpha)

plt.figure(figsize=(4, 3))
plt.axhline(y=0, xmin=0, xmax=len(W_trajectory_rep), linewidth=1, linestyle='--', color='gray')
plt.axvline(x=len(W_trajectory_rep) / 2, linewidth=1, linestyle='--', color='gray')
plt.plot(alpha_trajectory, color='black', lw=2, label=r'$\alpha$')
plt.plot(W_trajectory_rep, 'o-', ms=6, color='navy', lw=2, label=r'$w$')
plt.plot(Theta_trajectory_rep, 'o-', ms=6, color='crimson', lw=2, label=r'$\theta$')
plt.xlim(0, len(W_trajectory_rep) - 1)
plt.legend()

plt.xlabel('Curriculum step', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig_ind = 1
plt.tight_layout()
plt.savefig('curriculum.png', transparent=True, bbox_inches='tight', dpi=600)

plt.show()
