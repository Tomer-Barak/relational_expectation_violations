import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from setuptools.command.rotate import rotate


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


def create_single_quiver_plot(alpha_from, alpha_to, lambda_=0.5, r=1, w_range_max=1, theta_range_max=1):
    """Creates a single quiver plot with trajectory for given alpha values"""
    figure = plt.figure(figsize=(5, 5))
    w_range = np.linspace(-w_range_max, w_range_max, 20)
    theta_range = np.linspace(-theta_range_max, theta_range_max, 20)
    W, Theta = np.meshgrid(w_range, theta_range)
    W_dot = dz_dot(W, Theta, alpha_to, lambda_, r, k=1, gamma=1)
    Theta_dot = theta_dot_of_dz(W, Theta, alpha_to, lambda_, r, gamma=1)

    # Normalize the vectors
    magnitude = np.sqrt(W_dot ** 2 + Theta_dot ** 2)
    W_dot = W_dot / magnitude
    Theta_dot = Theta_dot / magnitude

    # nullclines
    w_vals = np.linspace(-w_range_max, w_range_max, 200)
    theta_vals = np.linspace(-theta_range_max, theta_range_max, 200)
    W_prime, Theta_prime = np.meshgrid(w_vals, theta_vals)
    Z = dz_dot(W_prime, Theta_prime, alpha_to, lambda_, r, k=1, gamma=1)
    Z_ = theta_dot_of_dz(W_prime, Theta_prime, alpha_to, lambda_, r, gamma=1)
    plt.contour(W_prime, Theta_prime, Z, levels=[0], colors='navy')
    plt.contour(W_prime, Theta_prime, Z_, levels=[0], colors='crimson')

    # Quiver plot with improved styling
    plt.quiver(W, Theta, W_dot, Theta_dot, color='black', angles='xy',
               pivot='mid', width=0.01, headwidth=5, headlength=5, headaxislength=3)  # , scale=100)

    # Trajectory plot
    W_trajectory = [r / np.sqrt(2) * (alpha_to / alpha_from)]
    Theta_trajectory = [r / np.sqrt(2)]
    plt.plot(W_trajectory[0], Theta_trajectory[0], 'o', color='gold', ms=10, markeredgecolor='teal')

    if alpha_to != alpha_from:
        dt = 0.0001
        T = 1000
        epsilon = 1e-4

        # D = []

        for t in range(1, int(T / dt) + 1):
            W_dot_t = dz_dot(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, k=1, gamma=1)
            Theta_dot_t = theta_dot_of_dz(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, gamma=1)

            # D.append(W_dot_t + Theta_dot_t)

            W_trajectory.append(W_trajectory[-1] + W_dot_t * dt)
            Theta_trajectory.append(Theta_trajectory[-1] + Theta_dot_t * dt)

            if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
                break

        # plt.plot(D)
        # plt.show()
        # exit()

        plt.plot(W_trajectory, Theta_trajectory, color='gold', lw=3)

        # Add arrows to the trajectory
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


    plt.title(r'$\alpha$=' + f'{alpha_from}', fontsize=16)
    plt.xlabel(r'$\Delta Z$', fontsize=16)
    ylabel = plt.ylabel(r'$\theta$', fontsize=16)
    plt.xticks(fontsize=14, rotation=45)
    # plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ylabel.set_rotation(0)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.ylim(-theta_range_max * 1.1, theta_range_max * 1.1)
    plt.xlim(-w_range_max * 1.1, w_range_max * 1.1)
    plt.tight_layout()
    plt.savefig(f'w_theta_quiver_alpha_from={alpha_from}_alpha_to={alpha_to}_lambda={lambda_}_r={r}.png',
                transparent=True, bbox_inches='tight', dpi=400)

    return figure


def plot_different_alpha_tos(lambda_=0.5, r=1):
    """Creates a figure comparing different alpha_to values with alpha_from=alpha_to"""
    fig = plt.figure(figsize=(5, 5))
    w_range_max = 3
    theta_range_max = 2

    colors = ['navy', 'crimson']
    alpha_tos = [-0.5, -2.0]

    # Plot both quivers and nullclines with their corresponding colors
    w_range = np.linspace(-w_range_max, w_range_max, 20)
    theta_range = np.linspace(-theta_range_max, theta_range_max, 20)
    W, Theta = np.meshgrid(w_range, theta_range)

    for alpha_to, color in zip(alpha_tos, colors):
        # Setup quiver
        W_dot = dz_dot(W, Theta, alpha_to, lambda_, r, k=1, gamma=1)
        Theta_dot = theta_dot_of_dz(W, Theta, alpha_to, lambda_, r, gamma=1)

        # Normalize vectors
        magnitude = np.sqrt(W_dot ** 2 + Theta_dot ** 2)
        W_dot = W_dot / magnitude
        Theta_dot = Theta_dot / magnitude

        # Plot quiver with corresponding color
        plt.quiver(W, Theta, W_dot, Theta_dot, color=color, angles='xy',
                   pivot='mid', width=0.01, headwidth=5, headlength=5, headaxislength=3, alpha=0.7)

        # Plot nullclines
        w_vals = np.linspace(-w_range_max, w_range_max, 200)
        theta_vals = np.linspace(-theta_range_max, theta_range_max, 200)
        W_prime, Theta_prime = np.meshgrid(w_vals, theta_vals)
        Z = dz_dot(W_prime, Theta_prime, alpha_to, lambda_, r, k=1, gamma=1)
        Z_ = theta_dot_of_dz(W_prime, Theta_prime, alpha_to, lambda_, r, gamma=1)
        plt.contour(W_prime, Theta_prime, Z, levels=[0], colors='gray', alpha=0.2)
        plt.contour(W_prime, Theta_prime, Z_, levels=[0], colors='gray', alpha=0.2)

    # Plot trajectories for different alpha_tos
    for alpha_to, color in zip(alpha_tos, colors):
        alpha_from = alpha_to  # Since we want alpha_from = alpha_to

        # Initial point - same as in plot_different_alpha_ratios
        W_trajectory = [-r / np.sqrt(2)]  # Since alpha_from = alpha_to, this simplifies to r/sqrt(2)
        Theta_trajectory = [r / np.sqrt(2)]
        plt.plot(W_trajectory[0], Theta_trajectory[0], 'o', color=color,
                 ms=10, markeredgecolor='black',
                 label=f'α={alpha_to}', alpha=0.6)

        # Calculate trajectory
        dt = 0.01
        T = 100
        epsilon = 1e-4
        for t in range(1, int(T / dt) + 1):
            W_dot_t = dz_dot(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, k=1, gamma=1)
            Theta_dot_t = theta_dot_of_dz(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, gamma=1)

            W_trajectory.append(W_trajectory[-1] + W_dot_t * dt)
            Theta_trajectory.append(Theta_trajectory[-1] + Theta_dot_t * dt)

            if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
                break

        plt.plot(W_trajectory, Theta_trajectory, color=color, lw=2)

        # Add arrows to the trajectory
        distances = np.sqrt(np.diff(W_trajectory) ** 2 + np.diff(Theta_trajectory) ** 2)
        cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
        total_distance = cumulative_distance[-1]
        num_arrows = 10
        interp_func_W_trajectory = interp1d(cumulative_distance, W_trajectory)
        interp_func_y = interp1d(cumulative_distance, Theta_trajectory)
        arrow_distances = np.linspace(0, total_distance, num_arrows + 1)
        arrow_W_trajectory = interp_func_W_trajectory(arrow_distances)
        arrow_y = interp_func_y(arrow_distances)
        for i in range(num_arrows):
            plt.annotate('', xy=(arrow_W_trajectory[i + 1], arrow_y[i + 1]),
                         xytext=(arrow_W_trajectory[i], arrow_y[i]),
                         arrowprops=dict(arrowstyle='->', color=color, lw=1))

    plt.xlabel(r'$\Delta Z$', fontsize=14)
    ylabel = plt.ylabel(r'$\theta$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ylabel.set_rotation(0)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.ylim(-theta_range_max * 1.1, theta_range_max * 1.1)
    plt.xlim(-w_range_max * 1.1, w_range_max * 1.1)
    # plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()

    plt.savefig(f'comparison_alpha_tos_lambda={lambda_}_r={r}.png',
                transparent=True, bbox_inches='tight', dpi=400)
    return fig


def plot_different_alpha_ratios(lambda_=0.5, r=1):
    """Creates a figure comparing different alpha_from/alpha_to ratios with fixed alpha_to=-1"""
    fig = plt.figure(figsize=(5, 5))
    w_range_max = 3
    theta_range_max = 2
    alpha_to = -1

    ratios = [0.5, 2.0]  # alpha_from/alpha_to ratios
    colors = ['navy', 'crimson', 'darkgreen']

    # Plot nullclines for background (using alpha_to)
    w_vals = np.linspace(-w_range_max, w_range_max, 200)
    theta_vals = np.linspace(-theta_range_max, theta_range_max, 200)
    W_prime, Theta_prime = np.meshgrid(w_vals, theta_vals)
    Z = dz_dot(W_prime, Theta_prime, alpha_to, lambda_, r, k=1, gamma=1)
    Z_ = theta_dot_of_dz(W_prime, Theta_prime, alpha_to, lambda_, r, gamma=1)
    plt.contour(W_prime, Theta_prime, Z, levels=[0], colors='gray', alpha=0.3)
    plt.contour(W_prime, Theta_prime, Z_, levels=[0], colors='gray', alpha=0.3)

    # Plot quiver for background (using alpha_to)
    w_range = np.linspace(-w_range_max, w_range_max, 20)
    theta_range = np.linspace(-theta_range_max, theta_range_max, 20)
    W, Theta = np.meshgrid(w_range, theta_range)
    W_dot = dz_dot(W, Theta, alpha_to, lambda_, r, k=1, gamma=1)
    Theta_dot = theta_dot_of_dz(W, Theta, alpha_to, lambda_, r, gamma=1)

    # Normalize vectors
    magnitude = np.sqrt(W_dot ** 2 + Theta_dot ** 2)
    W_dot = W_dot / magnitude
    Theta_dot = Theta_dot / magnitude

    # Plot background quiver with improved styling
    plt.quiver(W, Theta, W_dot, Theta_dot, color='black', angles='xy',
               pivot='mid', width=0.01, headwidth=5, headlength=5, headaxislength=3, alpha=0.7)

    # Plot trajectories for different ratios
    for ratio, color in zip(ratios, colors):
        alpha_from = -ratio * alpha_to  # since alpha_to is negative

        # Initial point
        W_trajectory = [r / np.sqrt(2) * (alpha_to / alpha_from)]
        Theta_trajectory = [r / np.sqrt(2)]
        plt.plot(W_trajectory[0], Theta_trajectory[0], 'o', color=color,
                 ms=10, markeredgecolor='black',
                 label=f'α_from/α_to={ratio}')

        # Calculate trajectory
        dt = 0.01
        T = 100
        epsilon = 1e-4
        for t in range(1, int(T / dt) + 1):
            W_dot_t = dz_dot(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, k=1, gamma=1)
            Theta_dot_t = theta_dot_of_dz(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, gamma=1)

            W_trajectory.append(W_trajectory[-1] + W_dot_t * dt)
            Theta_trajectory.append(Theta_trajectory[-1] + Theta_dot_t * dt)

            if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
                break

        plt.plot(W_trajectory, Theta_trajectory, color=color, lw=2)

        # Add arrows to the trajectory
        distances = np.sqrt(np.diff(W_trajectory) ** 2 + np.diff(Theta_trajectory) ** 2)
        cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
        total_distance = cumulative_distance[-1]
        num_arrows = 10
        interp_func_W_trajectory = interp1d(cumulative_distance, W_trajectory)
        interp_func_y = interp1d(cumulative_distance, Theta_trajectory)
        arrow_distances = np.linspace(0, total_distance, num_arrows + 1)
        arrow_W_trajectory = interp_func_W_trajectory(arrow_distances)
        arrow_y = interp_func_y(arrow_distances)
        for i in range(num_arrows):
            plt.annotate('', xy=(arrow_W_trajectory[i + 1], arrow_y[i + 1]),
                         xytext=(arrow_W_trajectory[i], arrow_y[i]),
                         arrowprops=dict(arrowstyle='->', color=color, lw=1))

    # plt.title(f'Different α ratios (α_to={alpha_to})', fontsize=14)
    plt.xlabel(r'$\Delta Z$', fontsize=14)
    ylabel = plt.ylabel(r'$\theta$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ylabel.set_rotation(0)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.ylim(-theta_range_max * 1.1, theta_range_max * 1.1)
    plt.xlim(-w_range_max * 1.1, w_range_max * 1.1)
    # plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()

    plt.savefig(f'quiver_different_alpha_ratios_alpha_to={alpha_to}_lambda={lambda_}_r={r}.png',
                transparent=True, bbox_inches='tight', dpi=400)
    return fig


def analyze_trajectory_crossing(alpha_from, alpha_to, lambda_=0.5, r=1):
    """Analyze whether a trajectory crosses theta=0 or Delta Z=0 first"""
    # Initial point
    W_trajectory = [r / np.sqrt(2) * (alpha_to / alpha_from)]
    Theta_trajectory = [r / np.sqrt(2)]

    dt = 0.0001
    T = 1000
    epsilon = 1e-4

    # # Keep track of previous point to detect crossing
    # prev_w = W_trajectory[0]
    # prev_theta = Theta_trajectory[0]
    #
    # for t in range(1, int(T / dt) + 1):
    #     W_dot_t = dz_dot(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, k=1, gamma=1)
    #     Theta_dot_t = theta_dot_of_dz(W_trajectory[-1], Theta_trajectory[-1], alpha_to, lambda_, r, gamma=1)
    #
    #     curr_w = W_trajectory[-1] + W_dot_t * dt
    #     curr_theta = Theta_trajectory[-1] + Theta_dot_t * dt
    #
    #     # Check for axis crossings
    #     if (prev_theta * curr_theta <= 0) and (abs(prev_theta) + abs(curr_theta) > 1e-10):  # crosses theta=0
    #         return 'theta'
    #     if (prev_w * curr_w <= 0) and (abs(prev_w) + abs(curr_w) > 1e-10):  # crosses dz=0
    #         return 'dz'
    #
    #     W_trajectory.append(curr_w)
    #     Theta_trajectory.append(curr_theta)
    #
    #     prev_w = curr_w
    #     prev_theta = curr_theta
    #
    #     if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
    #         break

    prev_w = W_trajectory[0]
    prev_theta = Theta_trajectory[0]

    solver = "euler"  # Choose between "euler" and "rk4"
    num_steps = int(T / dt)

    for _ in range(1, num_steps + 1):
        w_curr = W_trajectory[-1]
        theta_curr = Theta_trajectory[-1]

        # Compute derivatives
        W_dot_t = dz_dot(w_curr, theta_curr, alpha_to, lambda_, r, k=1, gamma=1)
        Theta_dot_t = theta_dot_of_dz(w_curr, theta_curr, alpha_to, lambda_, r, gamma=1)

        if solver == "euler":
            # Euler Method
            curr_w = w_curr + W_dot_t * dt
            curr_theta = theta_curr + Theta_dot_t * dt

        elif solver == "rk4":
            # Runge-Kutta 4th Order (RK4)
            def f_W(w, theta):
                return dz_dot(w, theta, alpha_to, lambda_, r, k=1, gamma=1)

            def f_Theta(w, theta):
                return theta_dot_of_dz(w, theta, alpha_to, lambda_, r, gamma=1)

            k1_w = dt * f_W(w_curr, theta_curr)
            k1_theta = dt * f_Theta(w_curr, theta_curr)

            k2_w = dt * f_W(w_curr + 0.5 * k1_w, theta_curr + 0.5 * k1_theta)
            k2_theta = dt * f_Theta(w_curr + 0.5 * k1_w, theta_curr + 0.5 * k1_theta)

            k3_w = dt * f_W(w_curr + 0.5 * k2_w, theta_curr + 0.5 * k2_theta)
            k3_theta = dt * f_Theta(w_curr + 0.5 * k2_w, theta_curr + 0.5 * k2_theta)

            k4_w = dt * f_W(w_curr + k3_w, theta_curr + k3_theta)
            k4_theta = dt * f_Theta(w_curr + k3_w, theta_curr + k3_theta)

            curr_w = w_curr + (k1_w + 2 * k2_w + 2 * k3_w + k4_w) / 6
            curr_theta = theta_curr + (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6

        else:
            raise ValueError("Unsupported solver. Choose 'euler' or 'rk4'.")

        # Check for axis crossings
        if (prev_theta * curr_theta <= 0) and (abs(prev_theta) + abs(curr_theta) > 1e-10):
            return 'theta'
        if (prev_w * curr_w <= 0) and (abs(prev_w) + abs(curr_w) > 1e-10):
            return 'dz'

        # Append new values
        W_trajectory.append(curr_w)
        Theta_trajectory.append(curr_theta)

        # Update previous values
        prev_w = curr_w
        prev_theta = curr_theta

        # Stopping condition
        if np.abs(W_dot_t) < epsilon and np.abs(Theta_dot_t) < epsilon:
            break

    return 'unknown'


def plot_crossing_phase_diagram(lambda_=0.1, r=1):
    """Create a phase diagram showing which axis gets crossed first for different alpha combinations"""
    # Generate parameter ranges
    alpha_froms = np.linspace(0.1, 3, 27)
    alpha_tos = np.linspace(-3, -0.1, 27)

    # Arrays to store results
    results = []

    # Run simulations
    for alpha_from in alpha_froms:

        for alpha_to in alpha_tos:
            crossing = analyze_trajectory_crossing(alpha_from, alpha_to, lambda_, r)
            results.append((alpha_from, alpha_to, crossing))

    # Separate results by crossing type
    theta_crossings = [(x, -y) for x, y, c in results if c == 'theta']
    dz_crossings = [(x, -y) for x, y, c in results if c == 'dz']

    # Create plot
    fig = plt.figure(figsize=(5, 5))

    if theta_crossings:
        theta_x, theta_y = zip(*theta_crossings)
        plt.scatter(theta_x, theta_y, marker='^', c='navy', s=50, label='θ=0')

    if dz_crossings:
        dz_x, dz_y = zip(*dz_crossings)
        plt.scatter(dz_x, dz_y, marker='s', c='crimson', s=50, label='ΔZ=0')

    plt.xlabel(r'$\alpha_1$', fontsize=16)
    ylabel = plt.ylabel(r'$\alpha_2$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # ylabel.set_rotation(0)
    plt.grid(False)
    # plt.axhline(0, color='black', lw=1)
    # plt.axvline(0, color='black', lw=1)

    # Add a dividing line y = -x to show where alpha_from = -alpha_to
    # x_line = np.linspace(0, 2, 100)
    plt.plot(alpha_froms, alpha_froms, '--', color='black', lw=2, label='α$_{from}$ = α$_{to}$')

    plt.plot(alpha_froms, 1 / alpha_froms, '-', color='black', lw=2, label=r'$\alpha_1\alpha_2$ = 1')

    plt.ylim(0, 3.1)
    plt.xlim(0, 3.1)

    # plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()

    plt.savefig(f'crossing_phase_diagram_lambda={lambda_}_r={r}.png',
                transparent=True, bbox_inches='tight', dpi=400)
    return fig


if __name__ == '__main__':
    # Original single plot
    # fig = create_single_quiver_plot(alpha_from=0.5, alpha_to=-0.5, lambda_=0.1)
    # plt.show()

    # New comparison plots
    fig_alpha_tos = plot_different_alpha_tos()
    plt.show()

    # fig_alpha_ratios = plot_different_alpha_ratios()
    # plt.show()

    # Phase diagram plot
    # fig_phase = plot_crossing_phase_diagram(lambda_=0.1)
    # plt.show()
