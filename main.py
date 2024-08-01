import itertools
import os, sys, ast
import time
from training import optimize
import numpy as np
import torch
import Images as imgs
import networks
import matplotlib.pyplot as plt
import plots
from torch.utils.tensorboard import SummaryWriter
import inspect


def parse_arguments(HP):
    dir_path = inspect.currentframe().f_back.f_code.co_name
    if len(sys.argv) > 1:
        parsed_args = {}
        for arg in sys.argv[1:]:
            key_value_pair = arg.split('=', 1)
            if len(key_value_pair) == 2:
                key, value = key_value_pair
                parsed_args[key] = value

        for key, value in parsed_args.items():
            dir_path += f"_{key}={value}"
            if key == 'rules':
                value = ast.literal_eval(value)
                for r_ind, r_key in enumerate(HP['rules'].keys()):
                    HP['rules'][r_key] = value[r_ind]
            elif key in ['total_alphas', 'eta_ratios', 'betas', 'gammas', 'circle_regularization', 'with_neg_betas']:
                HP[key] = eval(value)
            else:
                HP[key] = type(HP[key])(value)
    else:
        dir_path += f"_rules={tuple(HP['rules'].values())}_defaultArgs"

    if not os.path.exists(dir_path):
        if "[" in dir_path:
            dir_path = dir_path.replace('[', '_')
            dir_path = dir_path.replace(']', '_')
            dir_path = dir_path.replace(':', ',')
        os.makedirs(dir_path)

    return HP, dir_path


def create_nets(HP):
    Z_conv = networks.Z_conv(HP)

    temp_HP = HP.copy()
    temp_HP['batch'] = 2
    temp_HP['alpha'] = temp_HP['total_alpha']

    images = imgs.create_batch(temp_HP)
    n_input = Z_conv(images[0]).shape[1]

    Z_class = torch.nn.Linear(n_input, 1)
    Z = torch.nn.Sequential(Z_conv, Z_class)
    T = networks.T(HP)

    if HP['GPU']:
        Z, T = Z.cuda(), T.cuda()

    return Z, T


def gradual_rule_change(HP=None):
    internal = False
    if HP == None:
        internal = True
        print('gradual_rule_change (Internal)')
        HP = {'GPU': True,

              'grid_size': 224, 'channels': 3, 'plot_examples': False,
              'rules': {"color": 0,
                        "position": 0,
                        "size": 2,
                        "shape": 0,
                        "number": 0},

              'total_alpha': 0.8,
              'alpha_steps': 2,
              'alpha': None,
              'beta': None,
              'gamma': None,

              'batch': 2,
              'steps_per_batch': 20,
              'batches_per_alpha': 80,

              'lr': 4e-3, 'eta_ratio': 1., 'circle_regularization': 4,

              'dz_test_batch': 32, 'acc_test_batch': 32,

              'measure_optimization': True, 'only_for_RT': False, 'just_training': False,
              }

        if 'COLAB_GPU' in os.environ:
            input('gradual_rule_change ' + '1')
            HP, _ = parse_arguments(HP)

    Z, T = create_nets(HP)

    alphas = np.linspace(HP['total_alpha'], -HP['total_alpha'], HP['alpha_steps'])
    if HP['beta']:
        alphas = [alphas[0], HP['beta'], alphas[1]]
    if HP['just_training']:
        alphas = [alphas[0]]

    test_HP = HP.copy()
    test_HP['batch'] = HP['dz_test_batch']
    test_HP['alpha'] = HP['total_alpha']
    Z_test_images = imgs.create_batch(test_HP).cuda()

    total_measures = {}
    for alph_ind, alpha in enumerate(alphas):

        if internal:
            print('alpha =', alpha, flush=True)

        HP['alpha'] = alpha

        perf_HP = HP.copy()
        perf_HP['batch'] = HP['acc_test_batch']
        perf_test_images = imgs.create_batch(perf_HP).cuda()

        measures = optimize(HP, Z, T, Z_test_images, perf_test_images)

        if alph_ind > 0 and HP['measure_optimization'] and not HP['only_for_RT']:
            for key, val in measures.items():
                measures[key] = val[1:]

        if alph_ind == 0:
            total_measures = measures.copy()
        else:
            for key in total_measures.keys():
                total_measures[key] += measures[key]

    total_measures.update({'HP': HP})
    all_alphas = sum([[i] * HP['batches_per_alpha'] for i in alphas], [])
    total_measures.update({'Alphas': all_alphas})

    if internal and not HP['only_for_RT']:
        fname = f"internal_measures_alpha={HP['total_alpha']}_{np.random.randint(1000):04d}.pt"
        torch.save(total_measures, fname)
        plots.plot_measures(fname)

    return total_measures


def repeated_experiments():
    HP = {'GPU': True,

          'grid_size': 224, 'channels': 3, 'plot_examples': False,
          'rules': {"color": 0,
                    "position": 0,
                    "size": 2,
                    "shape": 0,
                    "number": 0},
          'total_alpha': 0.02,
          'alpha_steps': 2,
          'alpha': None,
          'beta': None,
          'gamma': None,
          'eta_ratio': 1.,

          'batch': 2,
          'steps_per_batch': 20,
          'batches_per_alpha': 80,

          'lr': 4e-3, 'circle_regularization': 4,

          'nets_per_total_alpha': 100, 'total_alphas': np.linspace(0.1, 1, 18),
          'eta_ratios': [1.], 'betas': [1.], 'gammas': [1.],
          'dz_test_batch': 32, 'acc_test_batch': 32,

          'measure_optimization': False, 'only_for_RT': False, 'just_training': False, 'with_neg_betas': False,
          }

    if 'COLAB_GPU' in os.environ:
        input('repeated_experiments ' + '4')
        pass

    HP, dir_path = parse_arguments(HP)

    if HP['with_neg_betas'] and len(HP['betas']) > 1:
        HP['betas'] = np.sort(np.concatenate([HP['betas'], -HP['betas']]))

    for gamma in HP['gammas']:
        for beta in HP['betas']:
            for eta_ratio in HP['eta_ratios']:
                for total_alpha in HP['total_alphas']:

                    HP['total_alpha'] = total_alpha

                    add_string = ""
                    if len(HP['eta_ratios']) > 1:
                        HP['eta_ratio'] = eta_ratio
                        add_string = f"_eta_ratio={eta_ratio:.6f}"
                    elif len(HP['betas']) > 1:
                        HP['beta'] = beta
                        add_string = f"_beta={beta:.6f}"
                    elif len(HP['gammas']) > 1:
                        HP['gamma'] = gamma
                        add_string = f"_gamma={gamma:.6f}"

                    fname = dir_path + f"/alpha={HP['total_alpha']:.6f}" + add_string + ".pt"
                    print(fname, flush=True)
                    if os.path.exists(fname):
                        continue
                    all_results = []
                    for N in range(HP['nets_per_total_alpha']):
                        start = time.time()
                        print(N, flush=True)
                        total_measures = gradual_rule_change(HP)
                        if total_measures:
                            all_results.append(total_measures)
                        who_changed = []
                        if np.sign(total_measures['T_params'][0]) != np.sign(total_measures['T_params'][-1]):
                            who_changed += ['T']
                        if np.sign(total_measures['dz_s'][0]) != np.sign(total_measures['dz_s'][-1]):
                            who_changed += ['Z']
                        print(
                            f'time={time.time() - start:.1f}, changed={who_changed},'
                            f" accs={total_measures['perf_accs']}", flush=True)

                    all_results.append({'HP': HP})
                    torch.save(all_results, fname)


if __name__ == '__main__':
    # gradual_rule_change()

    repeated_experiments()
