import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# Define the differential equations
def w_dot(w, theta, alpha, lambda_, r, k, gamma):
    theta = theta * (1 / gamma)
    return k * (-1 * (w * alpha - theta) * alpha - 1 * lambda_ * (
            (w * alpha) ** 2 + theta ** 2 - r ** 2) * 2 * alpha ** 2 * w)


def theta_dot(w, theta, alpha, lambda_, r, gamma):
    theta = theta * (1 / gamma)
    return (w * alpha - theta) / gamma - lambda_ * ((w * alpha) ** 2 + theta ** 2 - r ** 2) * 2 * theta / gamma


def dz_dot(w, theta, alpha, lambda_, r, k, gamma):
    theta = theta * (1 / gamma)
    return k * (-1 * alpha ** 2 * (w - theta) - lambda_ * (
            w ** 2 + theta ** 2 - r ** 2) * 2 * alpha ** 2 * w)


def theta_dot_of_dz(w, theta, alpha, lambda_, r, gamma):
    theta = theta * (1 / gamma)
    return (w - theta) / gamma - lambda_ * (w ** 2 + theta ** 2 - r ** 2) * 2 * theta / gamma


if __name__ == '__main__':

    # Parameter values
    alphas = [2, 0.5]
    lambda_ = 0.5
    r = 1

    w_range_max = 2
    theta_range_max = 2

    for i, alpha in enumerate(alphas):
        figure = plt.figure(figsize=(5, 5))
        w_range = np.linspace(-w_range_max, w_range_max, 20)
        theta_range = np.linspace(-theta_range_max, theta_range_max, 20)
        W, Theta = np.meshgrid(w_range, theta_range)
        W_dot = dz_dot(W, Theta, alpha, lambda_, r, k=1, gamma=1)
        Theta_dot = theta_dot_of_dz(W, Theta, alpha, lambda_, r, gamma=1)

        # Normalize the vectors
        magnitude = np.sqrt(W_dot ** 2 + Theta_dot ** 2)
        W_dot = W_dot / magnitude
        Theta_dot = Theta_dot / magnitude

        # nullclines
        w_vals = np.linspace(-w_range_max, w_range_max, 200)
        theta_vals = np.linspace(-theta_range_max, theta_range_max, 200)
        W_prime, Theta_prime = np.meshgrid(w_vals, theta_vals)
        Z = dz_dot(W_prime, Theta_prime, alpha, lambda_, r, k=1, gamma=1)
        Z_ = theta_dot_of_dz(W_prime, Theta_prime, alpha, lambda_, r, gamma=1)
        plt.contour(W_prime, Theta_prime, Z, levels=[0], colors='navy')
        plt.contour(W_prime, Theta_prime, Z_, levels=[0], colors='crimson')

        # quiver plot
        plt.quiver(W, Theta, W_dot, Theta_dot, color='black', angles='xy', pivot='mid', width=0.01, headwidth=3)

        # Plot trajectory
        W_trajectory = [-r / (np.sqrt(2))]
        Theta_trajectory = [r / np.sqrt(2)]

        plt.plot(W_trajectory[0], Theta_trajectory[0], 'o', color='gold', ms=10, markeredgecolor='teal')

        W_final = np.sign(alpha ** 2 - 1) * r / np.sqrt(2)
        Theta_final = np.sign(alpha ** 2 - 1) * r / np.sqrt(2)
        dt = 0.01
        T = 100
        epsilon = 1e-4
        for t in range(1, int(T / dt) + 1):
            W_dot_t = dz_dot(W_trajectory[-1], Theta_trajectory[-1], alpha, lambda_, r, k=1, gamma=1)
            Theta_dot_t = theta_dot_of_dz(W_trajectory[-1], Theta_trajectory[-1], alpha, lambda_, r, gamma=1)

            W_trajectory.append(W_trajectory[-1] + W_dot_t * dt)
            Theta_trajectory.append(Theta_trajectory[-1] + Theta_dot_t * dt)

            if np.abs(W_trajectory[-1] - W_final) < epsilon and np.abs(
                    Theta_trajectory[-1] - Theta_final) < epsilon:
                break

        plt.plot(W_trajectory, Theta_trajectory, color='gold', lw=3)

        # Plot arrows on the trajectory
        distances = np.sqrt(np.diff(W_trajectory) ** 2 + np.diff(Theta_trajectory) ** 2)
        cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
        total_distance = cumulative_distance[-1]
        num_arrows = 10
        interval_distance = total_distance / num_arrows
        interp_func_W_trajectory = interp1d(cumulative_distance, W_trajectory)
        interp_func_y = interp1d(cumulative_distance, Theta_trajectory)
        arrow_distances = np.linspace(0, total_distance, num_arrows + 1)
        arrow_W_trajectory = interp_func_W_trajectory(arrow_distances)
        arrow_y = interp_func_y(arrow_distances)

        for i in range(num_arrows):
            plt.annotate('', xy=(arrow_W_trajectory[i + 1], arrow_y[i + 1]), xytext=(arrow_W_trajectory[i], arrow_y[i]),
                         arrowprops=dict(arrowstyle='->', color='teal', lw=1))

        plt.title(r'$\alpha$ =' + f'{alpha}', fontsize=14)
        plt.xlabel(r'$\tilde{w}$', fontsize=14)
        ylabel = plt.ylabel(r'$\theta$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ylabel.set_rotation(0)
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.ylim(-theta_range_max * 1.1, theta_range_max * 1.1)
        plt.xlim(-w_range_max * 1.1, w_range_max * 1.1)

        plt.tight_layout()
        plt.savefig(f'w_theta_quiver_alpha={alpha}_lambda={lambda_}_r={r}.png', transparent=True,
                    bbox_inches='tight',
                    dpi=400)

    plt.show()
