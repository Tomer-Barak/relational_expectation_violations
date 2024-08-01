import time

from quiver_with_circ import theta_dot, w_dot, dz_dot, theta_dot_of_dz
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from scipy.interpolate import interp1d


def power_law(x, a, b):
    return a * x ** b


def simulate_trajectory(alpha_from, alpha_to, dt, T, k=1., beta=1., gamma=1., plot=False):
    Z_adapt = None
    converged = False
    epsilon = 1e-4

    W_0 = r / np.sqrt(2) * (alpha_to / alpha_from)
    Theta_0 = r / np.sqrt(2)

    W_trajectory = [W_0]
    Theta_trajectory = [Theta_0]

    if plot:
        plt.ion()
        # Plot the ellipse
        t = np.linspace(0, 2 * np.pi, 100)
        w_ellipse = (r / alpha) * np.cos(t)
        theta_ellipse = r * np.sin(t)
        plt.plot(w_ellipse, theta_ellipse, color='crimson', linewidth=1.5, alpha=0.6)

        w_line = np.linspace(-2, 2, 100)
        theta_line = alpha * w_line
        plt.plot(w_line, theta_line, 'teal', linewidth=1.5, alpha=0.6)

        plt.title(r'$\alpha$ =' + f'{alpha}', fontsize=14)
        plt.xlabel('w', fontsize=14)
        ylabel = plt.ylabel(r'$\theta$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ylabel.set_rotation(0)
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        # plt.ylim(-2.1, 2.1)
        # plt.xlim(-2.1, 2.1)

    while not converged:
        W_dot_t = dz_dot(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, k, gamma)
        Theta_dot_t = theta_dot_of_dz(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, gamma)
        W_trajectory.append(W_trajectory[-1] + W_dot_t * dt)
        Theta_trajectory.append(Theta_trajectory[-1] + Theta_dot_t * dt)

        if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
            converged = True

        if np.sign(W_trajectory[-1]) != np.sign(W_trajectory[0]):
            Z_adapt = 1
            return Z_adapt
        elif np.sign(Theta_trajectory[-1]) != np.sign(Theta_trajectory[0]):
            Z_adapt = 0
            return Z_adapt

        if plot:
            plt.plot(W_trajectory[-1], Theta_trajectory[-1], '.', ms=2, color='navy')
            # plt.xlim(-2, 2)
            # plt.ylim(-2, 2)
            plt.plot(w_ellipse, theta_ellipse, color='crimson', linewidth=1.5, alpha=0.6)
            plt.title(t)

            w_line = np.linspace(-2, 2, 100)
            theta_line = alpha * w_line
            plt.plot(w_line, theta_line, 'teal', linewidth=1.5, alpha=0.6)
            plt.pause(0.1)
            plt.clf()

    return Z_adapt


if __name__ == '__main__':
    lambda_ = 0.2
    r = 1

    ks = [1]  # np.logspace(-2, 2, 9, base=2)
    # betas = np.logspace(-2, 2, 9, base=2)
    betas = np.linspace(0.5, 2, 9)
    betas = np.sort(np.concatenate([-betas, betas]))
    gammas = [1]  # np.logspace(-1, 1, 9, base=2)

    manipulator = betas

    alphas = np.linspace(0.1, 8, 200)

    thresholds = []
    for k in ks:
        for gamma in gammas:
            for beta in betas:
                print(f'k = {k}, gamma = {gamma}, beta = {beta}')
                Z_adapts = []
                for alpha in alphas:
                    if beta < 0:
                        Z_adapt = simulate_trajectory(alpha_from=alpha, alpha_to=beta, dt=0.001, T=1000, k=k, beta=beta,
                                                      gamma=gamma, plot=False)
                    else:
                        Z_adapt = simulate_trajectory(alpha_from=beta, alpha_to=-alpha, dt=0.001, T=1000, k=k,
                                                      beta=beta,
                                                      gamma=gamma, plot=False)
                    Z_adapts.append(Z_adapt)

                Z_adapts = np.array(Z_adapts)
                if (Z_adapts == 1).any() and (Z_adapts == 0).any():
                    threshold = np.abs(alphas[np.where(Z_adapts > 0.5)[0][0]])
                else:
                    if (Z_adapts == 1).all():
                        threshold = 0
                    elif (Z_adapts == 0).all():
                        threshold = 10
                thresholds.append(threshold)

    thresholds = np.array(thresholds)
    interp_func = interp1d(manipulator, np.array(thresholds), kind='cubic')


    def f_interp(x):
        return interp_func(x) - 1


    zero_result = root_scalar(f_interp, bracket=[manipulator[0], 0.1], method='brentq')
    zero1 = zero_result.root
    zero_result = root_scalar(f_interp, bracket=[-0.1, manipulator[-1]], method='brentq')
    zero2 = zero_result.root

    plt.figure(figsize=(4, 3))
    plt.plot(manipulator, thresholds, 'o', ms=6, color='navy', zorder=10)

    xlims = plt.gca().get_xlim()
    # ylims = plt.gca().get_ylim()
    # plt.vlines(x=zero1, ymin=ylims[0], ymax=1, linestyle='dashed', color='gray', zorder=1)
    # plt.vlines(x=zero2, ymin=ylims[0], ymax=1, linestyle='dashed', color='gray', zorder=1)
    plt.hlines(y=1, xmin=xlims[0], xmax=xlims[1], linestyle='dashed', color='gray', zorder=1)
    plt.xlim(xlims)
    # plt.ylim(ylims)

    plt.ylabel(r'$\bar\alpha_{sim}$', fontsize=14)#, rotation=0, labelpad=10)
    plt.xlabel(r'$\beta$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('beta_alpha_bar.png', dpi=500, bbox_inches='tight', transparent=True)
    plt.show()
