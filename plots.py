import glob
import ast
import inspect
import re
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


def plot_measures(fname, mask='invert_dzs'):
    measures = torch.load(fname)

    if type(measures) == list:
        HP, Alphas, dz_means, dz_means_error, T_s, T_s_error, perf_accs, perf_accs_error, _ = average_measures(measures,
                                                                                                               mask)
    else:
        HP = measures['HP']
        Alphas = measures['Alphas']
        T_s = np.array(measures['T_params'])
        dz_means = np.array(measures['dz_s'])
        Mean_losses = measures['Mean_losses']
        perf_accs = measures['perf_accs']
        Alphas = [Alphas[0]] + Alphas

    epochs = np.arange(0, HP['batches_per_alpha'] * HP['alpha_steps'] + 1)

    fig, ax = plt.subplots(2, 2)
    plt.sca(ax[0, 0])
    plt.plot(epochs, Alphas, lw=2)
    plt.ylabel('Alpha', fontsize=14)

    plt.sca(ax[0, 1])
    plt.plot(epochs, T_s, lw=2, color='C0')
    plt.ylabel('T', fontsize=14)
    plt.ylim(-1.1 * np.max(np.abs(T_s)), 1.1 * np.max(np.abs(T_s)))

    plt.sca(ax[1, 0])
    plt.plot(epochs, dz_means, lw=2)
    plt.ylabel('dz', fontsize=14)
    plt.ylim(-1.1 * np.max(np.abs(dz_means)), 1.1 * np.max(np.abs(dz_means)))

    # plt.sca(ax[1, 1])
    # plt.plot(epochs, Mean_losses, lw=2)
    # plt.ylim((0, 1.1 * np.max(Mean_losses)))
    # plt.ylabel('Loss', fontsize=14)

    plt.sca(ax[1, 1])
    plt.plot(epochs, perf_accs, lw=2)
    plt.ylim((-0.1, 1.1))
    plt.ylabel('Accuracy', fontsize=14)

    plt.tight_layout()
    # plt.savefig(fname[:-3] + '.png', dpi=600)
    plt.show()


def average_measures(measures, mask):
    if type(measures) == list:
        HP = measures[-1]['HP']
        all_measures = {}
        for measure in measures[:-1]:
            for key, value in measure.items():
                if key not in all_measures:
                    all_measures[key] = [value]
                else:
                    all_measures[key].append(value)

        Alphas = [all_measures['Alphas'][0][0]] + all_measures['Alphas'][0]
        T_s = np.array(all_measures['T_params'])
        dz_means = np.array(all_measures['dz_s'])
        for ind_T, T in enumerate(T_s):
            if T[0] < 0:
                T_s[ind_T] = -T_s[ind_T]
                dz_means[ind_T] = -dz_means[ind_T]

        perf_accs = np.array(all_measures['perf_accs'])

        if mask == 'invert_dzs':
            ind_middle = HP['batches_per_alpha'] + 1
            dz_means[:, ind_middle:] = -dz_means[:, ind_middle:]

        # T_s_mean = np.mean(T_s, axis=0)
        T_s_mean = T_s[1, :]
        # T_s_error = 1.96 * st.sem(T_s, axis=0)
        T_s_error = np.zeros_like(T_s_mean)

        # dz_means_mean = np.mean(dz_means, axis=0)
        dz_means_mean = dz_means[1, :]
        # dz_means_error = 1.96 * st.sem(dz_means, axis=0)
        dz_means_error = np.zeros_like(dz_means_mean)

        perf_accs_mean = np.mean(perf_accs, axis=0)
        perf_accs_error = 1.96 * st.sem(perf_accs, axis=0)

    else:
        HP = measures['HP']
        Alphas = [measures['Alphas'][0]] + measures['Alphas']
        T_s_mean = np.array(measures['T_params'])
        T_s_error = np.zeros_like(T_s_mean)
        dz_means_mean = np.array(measures['dz_s'])
        dz_means_error = np.zeros_like(dz_means_mean)

        perf_accs_mean = measures['perf_accs']
        perf_accs_error = np.zeros_like(perf_accs_mean)
        perf_accs = np.array(measures['perf_accs'])

    return HP, Alphas, dz_means_mean, dz_means_error, T_s_mean, T_s_error, perf_accs_mean, perf_accs_error, perf_accs


def plot_two_alphas_together(folder, mask=None):
    fname1 = folder + '/alpha=0.800000.pt'
    fname2 = folder + '/alpha=0.200000.pt'

    rules = folder[folder.find('rules=') + len('rules='):folder.find('rules=') + len('rules=') + len('(0,0,0,0,0)')]
    rules = ast.literal_eval(rules)
    possible_rules = ['color', '', 'size', '', 'number']
    rule = possible_rules[rules.index(2)]

    measures1 = torch.load(fname1, weights_only=False)
    HP1, Alphas1, dz_means_mean1, dz_means_error1, T_s_mean1, T_s_error1, perf_accs_mean1, perf_accs_error1, _ = average_measures(
        measures1, mask)

    measures2 = torch.load(fname2, weights_only=False)
    HP2, Alphas2, dz_means_mean2, dz_means_error2, T_s_mean2, T_s_error2, perf_accs_mean2, perf_accs_error2, _ = average_measures(
        measures2, mask)

    epochs = np.arange(0, HP1['batches_per_alpha'] * HP1['alpha_steps'] + 1) * HP1['batch']
    middle_epoch_ind = HP1['batches_per_alpha'] + 1

    color1 = 'navy'
    color2 = 'crimson'


    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0.5, color='black', linestyle='--', lw=1)
    plt.plot(epochs, perf_accs_mean1, lw=2, color=color1)
    plt.fill_between(epochs, perf_accs_mean1 - perf_accs_error1, perf_accs_mean1 + perf_accs_error1, alpha=0.3,
                     color=color1)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Example pairs', fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1] + '_large_alpha_accs.png', transparent=True, bbox_inches='tight', dpi=600)

    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0.5, color='black', linestyle='--', lw=1)
    plt.plot(epochs, perf_accs_mean2, lw=2, color=color1)
    plt.fill_between(epochs, perf_accs_mean2 - perf_accs_error2, perf_accs_mean2 + perf_accs_error2, alpha=0.3,
                     color=color1)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Example pairs', fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1] + '_small_alpha_accs.png', transparent=True, bbox_inches='tight', dpi=600)

    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.plot(epochs, dz_means_mean1, lw=2, color=color2, label=r'$\Delta Z$')
    # plt.fill_between(epochs, dz_means_mean1 - dz_means_error1, dz_means_mean1 + dz_means_error1, alpha=0.3,
    #                  color=color1)
    plt.plot(epochs, T_s_mean1, lw=2, color=color1, label=r'$\theta$')
    # plt.fill_between(epochs, T_s_mean1 - T_s_error1, T_s_mean1 + T_s_error1, alpha=0.3,
    #                  color=color1)
    plt.legend(fontsize=12, loc='lower right')
    plt.ylabel(r'$\Delta z$', fontsize=14, rotation=0, labelpad=10)
    plt.xlabel('Example pairs', fontsize=12)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1] + '_large_alpha_two_alphas_dz_theta.png', transparent=True, bbox_inches='tight',
                dpi=600)

    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.plot(epochs, dz_means_mean2, lw=2, color=color2, label=r'$\Delta Z$')
    # plt.fill_between(epochs, dz_means_mean1 - dz_means_error1, dz_means_mean1 + dz_means_error1, alpha=0.3,
    #                  color=color1)
    plt.plot(epochs, T_s_mean2, lw=2, color=color1, label=r'$\theta$')
    # plt.fill_between(epochs, T_s_mean1 - T_s_error1, T_s_mean1 + T_s_error1, alpha=0.3,
    #                  color=color1)
    plt.legend(fontsize=12, loc='lower right')
    plt.ylabel(r'$\Delta z$', fontsize=14, rotation=0, labelpad=10)
    plt.xlabel('Example pairs', fontsize=12)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1] + '_small_alpha_two_alphas_dz_theta.png', transparent=True, bbox_inches='tight',
                dpi=600)

    # plt.figure(figsize=(4, 3))
    # plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    # plt.axhline(y=0, color='black', linestyle='--', lw=1)
    #
    #
    # plt.plot(epochs, T_s_mean2, lw=2, color=color1, label=r'$\theta$')
    # # plt.fill_between(epochs, T_s_mean2 - T_s_error2, T_s_mean2 + T_s_error2, alpha=0.3,
    # #                  color=color2)
    # plt.plot(epochs, dz_means_mean2, lw=2, color=color2, label=r'$\Delta Z$')
    # # plt.fill_between(epochs, dz_means_mean2 - dz_means_error2, dz_means_mean2 + dz_means_error2, alpha=0.3,
    # #                  color=color2)
    # plt.ylabel(r'$\theta$', fontsize=14, rotation=0, labelpad=10)
    # plt.xlabel('Example pairs', fontsize=12)
    # plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    # plt.legend(fontsize=12, loc='upper right')
    # plt.yticks(fontsize=12)
    # plt.ylim(-0.3, 0.3)
    # plt.tight_layout()
    # plt.savefig(folder.split('/')[1] + '_small_alpha_two_alphas_dz_theta.png', transparent=True, bbox_inches='tight', dpi=600)

    plt.show()


def two_fixed_points(folder, mask=None, net=0):
    folder = r'results\res_rules=(0,0,2,0,0)_just_training=True_measure_optimization=True_total_alphas=_0.5__nets_per_total_alpha=100'
    fname1 = folder + '/alpha=0.500000.pt'

    rules = folder[folder.find('rules=') + len('rules='):folder.find('rules=') + len('rules=') + len('(0,0,0,0,0)')]
    rules = ast.literal_eval(rules)
    possible_rules = ['color', '', 'size', '', 'number']
    rule = possible_rules[rules.index(2)]

    measures = torch.load(fname1, weights_only=False)
    HP = measures[-1]['HP']
    epochs = np.arange(0, HP['batches_per_alpha'] + 1) * HP['batch']

    color1 = 'navy'
    color2 = 'crimson'

    plt.figure(figsize=(4, 3))
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.plot(epochs, measures[net]['T_params'], lw=2, color=color1, label=r'$\theta$')
    plt.plot(epochs, measures[net]['dz_s'], lw=2, color=color2, label=r'$\Delta Z$')
    plt.xlabel('Example pairs', fontsize=12)
    #
    plt.xticks([0, 40, 80, 120, 160], fontsize=12)
    plt.ylim(-0.3, 0.3)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'net{net}_fixed_points.png', transparent=True, bbox_inches='tight', dpi=600)
    plt.show()

def concept_rule_ratio_for_ANN_a1_to_a2(dir, eta_ratio, beta, gamma, gamma_in, mask):
    if eta_ratio:
        fnames = glob.glob(dir + f'\*_eta_ratio={eta_ratio:.6f}.pt')
    elif beta:
        fnames = glob.glob(dir + f'\*_beta={beta:.6f}.pt')
    elif gamma:
        fnames = glob.glob(dir + f'\*_gamma={gamma:.6f}.pt')
    elif gamma_in:
        fnames = glob.glob(dir + f'\*_gamma={gamma_in:.6f}.pt')
    else:
        fnames = glob.glob(dir + '\*.pt')

    ratios = []
    ratios_errs = []
    yields = []
    fname_alphas = []
    valid_alphas = []
    yield_errs = []
    RT_mean = []
    RT_error = []
    RTs_list = []
    final_perf_accs_mean, final_perf_accs_err = [], []
    for fname in fnames:
        all_measures = torch.load(fname, weights_only=False)

        if mask == 'alpha_from_0.1':
            if all_measures[-1]['HP']['total_alpha'] < 0.1:
                continue

        # yield_errs.append(1.96 * np.sqrt((yields[-1] * (1 - yields[-1])) / 49))
        # print('**************** Temporary use of yields num = 49 ************************')
        if eta_ratio:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_eta_ratio=')]))
        elif beta:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_beta=')]))
        elif gamma:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_gamma=')]))
        elif gamma_in:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_gamma=')]))
        else:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('.pt')]))

        all_measures = all_measures[:-1]

        concept_changes = 0
        rule_changes = 0
        RTs = []
        yld = 0
        final_perf_accs = []
        for meas in all_measures:
            dz_s = np.array(meas['dz_s'])
            T_params = np.array(meas['T_params'])
            HP = meas['HP']
            perf_accs = np.array(meas['perf_accs'])

            if mask == 'above_67':
                if HP['only_for_RT']:
                    if len(perf_accs) < (HP['batches_per_alpha'] * 2 + 2) and perf_accs[-1] > 0.67:
                        yld += 1
                    else:
                        continue
                else:
                    if np.all(perf_accs > 0.67):
                        yld += 1
                    else:
                        continue

            elif mask == 'not_above_67':
                if len(perf_accs) == 2:
                    if ~np.all(perf_accs > 0.67):
                        yld += 1
                    else:
                        continue
                else:
                    if perf_accs[-1] < 0.67 or perf_accs[int(len(perf_accs) / 2) - 1] < 0.67:
                        yld += 1
                    else:
                        continue

            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            caller_name = caller_frame.f_code.co_name
            if caller_name == 'acc_vs_alpha':
                ind = 0
            else:
                ind = -1
            final_perf_accs.append(perf_accs[ind])

            if HP['only_for_RT']:
                if len(meas['RT']) == 2:
                    RTs.append(meas['RT'])
                    continue
            else:
                if len(dz_s) == 2 or beta:
                    batches = 1
                else:
                    batches = HP['batches_per_alpha']

                if (np.sign(dz_s[batches - 1]) != np.sign(dz_s[-1]) and
                        np.sign(T_params[batches - 1]) == np.sign(T_params[-1])):
                    concept_changes += 1
                elif (np.sign(dz_s[batches - 1]) == np.sign(dz_s[-1]) and
                      np.sign(T_params[batches - 1]) != np.sign(T_params[-1])):
                    rule_changes += 1
                else:
                    print(fname)
                    print(40 * 'X' + ' Invalid network ' + 40 * 'X')
                    print(dz_s)
                    print(T_params)
                    print(perf_accs)
                    # exit()

        if mask == 'above_67' or mask == 'not_above_67':
            yields.append(np.copy(yld) / len(all_measures))
            yield_err = prop.proportion_confint(yields[-1] * len(all_measures), len(all_measures), method='wilson')
            yield_err = np.array(yield_err)
            yield_err[0] = np.array(yields[-1]) - yield_err[0]
            yield_err[1] = yield_err[1] - np.array(yields[-1])
            yield_errs.append(yield_err)
            if yields[-1] < 0.05:
                continue

        if HP['only_for_RT'] and len(all_measures) > 1:
            RTs = np.array(RTs)
            RTs_list.append(RTs)
            RT_mean.append(np.mean(RTs, axis=0))
            RT_error.append(1.96 * st.sem(RTs, axis=0))
            valid_alphas.append(HP['total_alpha'])
        elif not HP['only_for_RT']:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            caller_name = caller_frame.f_code.co_name
            if caller_name == 'acc_vs_alpha':
                final_perf_accs_mean.append(np.mean(final_perf_accs))
                final_perf_accs_err.append(1.96 * st.sem(final_perf_accs))
                valid_alphas.append(HP['total_alpha'])
                continue
            ratios.append(concept_changes / (concept_changes + rule_changes))
            # ratio_err = prop.proportion_confint(concept_changes, concept_changes + rule_changes, method='wilson')
            # ratio_err = np.array(ratio_err)
            # ratio_err[0] = np.array(ratios[-1]) - ratio_err[0]
            # ratio_err[1] = ratio_err[1] - np.array(ratios[-1])
            # ratios_errs.append(ratio_err)

            if len(all_measures) > 1:

                ratios_errs.append(
                    1.96 * np.sqrt((ratios[-1] * (1 - ratios[-1])) / (concept_changes + rule_changes - 1)))

            else:
                ratios_errs.append(0)
            valid_alphas.append(HP['total_alpha'])

    return fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, RT_mean, RT_error, RTs_list, final_perf_accs_mean, final_perf_accs_err


def concept_rule_ratio(dir, eta_ratio, beta, gamma, mask='alpha_from_0.1'):
    if eta_ratio:
        fnames = glob.glob(dir + f'\*_eta_ratio={eta_ratio:.6f}.pt')
    elif beta:
        fnames = glob.glob(dir + f'\*_beta={beta:.6f}.pt')
    elif gamma:
        fnames = glob.glob(dir + f'\*_gamma={gamma:.6f}.pt')
    else:
        fnames = glob.glob(dir + '\*.pt')

    ratios = []
    ratios_errs = []
    yields = []
    fname_alphas = []
    valid_alphas = []
    yield_errs = []
    RT_mean = []
    RT_error = []
    RTs_list = []
    final_perf_accs_mean, final_perf_accs_err = [], []
    excluded = 0
    for fname in fnames:
        all_measures = torch.load(fname, weights_only=False)

        if mask == 'alpha_from_0.1':
            if all_measures[-1]['HP']['total_alpha'] < 0.1:
                continue

        if eta_ratio:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_eta_ratio=')]))
        elif beta:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_beta=')]))
        elif gamma:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('_gamma=')]))
        else:
            fname_alphas.append(float(fname[fname.find('\\alpha=') + len('\\alpha='):fname.find('.pt')]))

        all_measures = all_measures[:-1]

        concept_changes = 0
        rule_changes = 0
        RTs = []

        final_perf_accs = []
        for meas in all_measures:
            dz_s = np.array(meas['dz_s'])
            T_params = np.array(meas['T_params'])
            HP = meas['HP']
            perf_accs = np.array(meas['perf_accs'])

            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            caller_name = caller_frame.f_code.co_name
            if caller_name == 'acc_vs_alpha':
                ind = 0
            else:
                ind = -1
            final_perf_accs.append(perf_accs[ind])

            if HP['only_for_RT']:
                if len(meas['RT']) == 2:
                    RTs.append(meas['RT'])
                    continue
            else:
                if len(dz_s) == 2 or beta:
                    batches = 1
                else:
                    batches = HP['batches_per_alpha']

                if (np.sign(dz_s[batches - 1]) != np.sign(dz_s[-1]) and
                        np.sign(T_params[batches - 1]) == np.sign(T_params[-1])):
                    concept_changes += 1
                elif (np.sign(dz_s[batches - 1]) == np.sign(dz_s[-1]) and
                      np.sign(T_params[batches - 1]) != np.sign(T_params[-1])):
                    rule_changes += 1
                else:
                    if 'gamma_ins' in HP.keys():
                        if len(HP['gamma_ins']) > 1:
                            if HP['total_alpha'] > 0.1:
                                excluded += 1
                    else:
                        excluded += 1

        if HP['only_for_RT'] and len(all_measures) > 1:
            RTs = np.array(RTs)
            RTs_list.append(RTs)
            RT_mean.append(np.mean(RTs, axis=0))
            RT_error.append(1.96 * st.sem(RTs, axis=0))
            valid_alphas.append(HP['total_alpha'])
        elif not HP['only_for_RT']:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            caller_name = caller_frame.f_code.co_name
            if caller_name == 'acc_vs_alpha':
                final_perf_accs_mean.append(np.mean(final_perf_accs))
                final_perf_accs_err.append(1.96 * st.sem(final_perf_accs))
                valid_alphas.append(HP['total_alpha'])
                continue
            ratios.append(concept_changes / (concept_changes + rule_changes))
            # print(fname, 'Excluded:', len(all_measures) - (concept_changes + rule_changes), 'out of', len(all_measures))
            if len(all_measures) > 1:

                ratios_errs.append(
                    1.96 * np.sqrt((ratios[-1] * (1 - ratios[-1])) / (concept_changes + rule_changes - 1)))

            else:
                ratios_errs.append(0)
            valid_alphas.append(HP['total_alpha'])

    print(f'{dir}\nexcluded {excluded} out of {len(all_measures) * 18}')

    return fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, RT_mean, RT_error, RTs_list, final_perf_accs_mean, final_perf_accs_err


def plot_concept_rule_ratios(fname_base, eta_ratio=None, beta=None, gamma=None):
    rules = ['(2,0,0,0,0)', '(0,0,2,0,0)', '(0,0,0,0,2)']

    dirs = [fname_base[:fname_base.find('rules=') + len('rules=')] + rule + fname_base[fname_base.find('rules=') + len(
        'rules=') + len(rule):] for rule in rules]

    fname_base = (fname_base[:fname_base.find('rules=') - 1] +
                  fname_base[fname_base.find('rules=') + len('rules=') + len(rules[0]):])

    fname_alphas3, valid_alphas3, yields3, yield_errs3, ratios3, ratios_errs3 = [], [], [], [], [], []
    for dir in dirs:
        fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, _, _, _, _, _ = concept_rule_ratio(dir,
                                                                                                                eta_ratio,
                                                                                                                beta,
                                                                                                                gamma)
        fname_alphas3.append(fname_alphas)
        valid_alphas3.append(valid_alphas)
        yields3.append(yields)
        yield_errs3.append(yield_errs)
        ratios3.append(ratios)
        ratios_errs3.append(ratios_errs)

    color = 'black'
    fit_color = 'teal'

    for i in range(3):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        # plt.sca(ax[i])

        guess = (1, 0.5)
        popt, pcov = curve_fit(sigmoid, valid_alphas3[i], ratios3[i], guess)
        plt.plot(valid_alphas3[i], 100 - 100 * sigmoid(valid_alphas3[i], *popt), color=fit_color)
        plt.axvline(popt[1], 0, 0.97, color='gray', linestyle='--', lw=1)
        plt.errorbar(valid_alphas3[i], 100 - 100 * np.array(ratios3[i]), yerr=100 * np.array(ratios_errs3[i]).T,
                     fmt='o',
                     color=color)

        plt.text(popt[1] - 0.08, 0., r'$\bar\alpha$', fontsize=14, ha='left', va='bottom',
                 transform=plt.gca().transData)
        print(rule, popt[1], 1.96 * np.sqrt(np.diag(pcov))[1])

        plt.ylim(-0.1 * 100, 1.1 * 100)
        plt.xlim(0, valid_alphas3[i][-1] + 0.1 * (valid_alphas3[i][-1] - valid_alphas3[i][0]))
        # plt.xscale('log')
        plt.ylabel(r'Fraction of $\theta$' + '\nchanging sign (%)', fontsize=14)
        if rule != 'size':
            plt.xlabel(r'$\alpha_{%s}$' % rule, fontsize=14)
        else:
            plt.xlabel(r'$\alpha$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + '_' + rule + '_concept_ratios.png', bbox_inches='tight',
                    transparent=True, dpi=500)

    plt.show()


def plot_betas(fname_base):
    rules = ['(2,0,0,0,0)', '(0,0,2,0,0)', '(0,0,0,0,2)']

    dirs = [fname_base[:fname_base.find('rules=') + len('rules=')] + rule + fname_base[fname_base.find('rules=') + len(
        'rules=') + len(rule):] for rule in rules]

    fname_base = (fname_base[:fname_base.find('rules=') - 1] +
                  fname_base[fname_base.find('rules=') + len('rules=') + len(rules[0]):])

    fname_alphas3, valid_alphas3, yields3, yield_errs3, ratios3, ratios_errs3, betas3 = [], [], [], [], [], [], []
    for dir in dirs:
        betas1 = eval(
            dir[dir.find('betas=') + len('betas='):[m.start() for m in re.finditer("\)", dir)][1] + 1])
        betas = np.sort(np.concatenate([betas1, -betas1]))
        fname_alphas_eta = []
        valid_alphas_eta = []
        yields_eta = []
        yield_errs_eta = []
        ratios_eta = []
        ratios_errs_eta = []
        for beta in betas:
            fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, _, _, _, _, _ = concept_rule_ratio(dir,
                                                                                                                    eta_ratio=None,
                                                                                                                    beta=beta,
                                                                                                                    gamma=None)
            fname_alphas_eta.append(fname_alphas)
            valid_alphas_eta.append(valid_alphas)
            yields_eta.append(yields)
            yield_errs_eta.append(yield_errs)
            ratios_eta.append(ratios)
            ratios_errs_eta.append(ratios_errs)
        fname_alphas3.append(fname_alphas_eta)
        valid_alphas3.append(valid_alphas_eta)
        yields3.append(yields_eta)
        yield_errs3.append(yield_errs_eta)
        ratios3.append(ratios_eta)
        ratios_errs3.append(ratios_errs_eta)
        betas3.append(betas)

    color = 'navy'

    alpha_stars3 = []
    alpha_stars_errs3 = []
    valid_betas3 = []
    for i in range(len(fname_alphas3)):
        alpha_stars_eta = []
        alpha_stars_errs_eta = []
        valid_betas_eta = []
        for j in range(len(fname_alphas3[0])):
            guess = (1, 0.5)
            try:
                popt, pcov = curve_fit(sigmoid, valid_alphas3[i][j], ratios3[i][j], guess)
                perr = np.sqrt(np.diag(pcov))
                if popt[1] >= 0:
                    alpha_stars_eta.append(popt[1])
                    alpha_stars_errs_eta.append(perr[1] * 1.96)
                    valid_betas_eta.append(betas3[i][j])
            except:
                pass
        valid_betas3.append(valid_betas_eta)

        alpha_stars3.append(alpha_stars_eta)
        alpha_stars_errs3.append(alpha_stars_errs_eta)

    for i in range(len(fname_alphas3)):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        plt.errorbar(np.array(valid_betas3[i]), (alpha_stars3[i]), yerr=alpha_stars_errs3[i], color=color, fmt='o')
        if rule != 'size':
            plt.ylabel(r'$\bar{\alpha}_{%s}$' % rule, fontsize=14)
        else:
            plt.ylabel(r'$\bar{\alpha}$', rotation=0, labelpad=10, fontsize=14)
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.xlim(-1.1, 1.1)
        plt.xticks(fontsize=12)
        xlims = plt.gca().get_xlim()
        if rule == 'size' or rule == 'color':
            alphabar = 0.34
        elif rule == 'number':
            alphabar = 0.35
        plt.hlines(y=alphabar, xmin=xlims[0], xmax=xlims[1], linestyle='dashed', color='gray', zorder=1)
        plt.xlim(xlims)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + f'_{rule}_betas.png', bbox_inches='tight', transparent=True, dpi=500)

    plt.show()


def plot_just_training(fname):
    measures1 = torch.load(fname, weights_only=False)
    (HP1, Alphas1, dz_means_mean1, dz_means_error1, T_s_mean1, T_s_error1, perf_accs_mean1, perf_accs_error1,
     perf_accs) = average_measures(measures1, mask='invert_dzs')

    epochs = np.arange(0, HP1['batches_per_alpha'] + 1) * HP1['batch']
    middle_epoch_ind = int(HP1['batches_per_alpha'] / 2)

    cut_index = np.shape(perf_accs)[1] - 1
    perf_accs_final = perf_accs[:, cut_index]
    print(f'Final accuracy = {np.mean(perf_accs_final):.2f}+-{1.96 * st.sem(perf_accs_final):.2f}\n'
          f'Yield = {sum(perf_accs_final > 0.67) / len(perf_accs_final):.2f}')

    plt.figure(figsize=(4, 3))
    color1 = 'navy'

    plt.axhline(y=0.5, color='black', linestyle='--', lw=1)
    plt.plot(epochs, perf_accs_mean1, lw=2, color=color1)
    plt.fill_between(epochs, perf_accs_mean1 - perf_accs_error1, perf_accs_mean1 + perf_accs_error1, alpha=0.3,
                     color=color1)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Example pairs', fontsize=12)

    plt.xticks(
        [0, epochs[int(middle_epoch_ind * 1 / 2)], epochs[middle_epoch_ind], epochs[int(middle_epoch_ind * 3 / 2)],
         epochs[-1]], fontsize=12)

    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(fname.split('/')[1] + '_just_training.png', transparent=True, bbox_inches='tight', dpi=600)
    plt.show()


def acc_vs_alpha(fname_base):
    rules = ['(2,0,0,0,0)', '(0,0,2,0,0)', '(0,0,0,0,2)']

    dirs = [fname_base[:fname_base.find('rules=') + len('rules=')] + rule + fname_base[fname_base.find('rules=') + len(
        'rules=') + len(rule):] for rule in rules]

    fname_base = (fname_base[:fname_base.find('rules=') - 1] +
                  fname_base[fname_base.find('rules=') + len('rules=') + len(rules[0]):])

    fname_alphas3, valid_alphas3, final_perf_accs_mean3, final_perf_accs_err3 = [], [], [], []

    for dir in dirs:
        fname_alphas, valid_alphas, _, _, _, _, _, _, _, final_perf_accs_mean, final_perf_accs_err = concept_rule_ratio(
            dir,
            eta_ratio=None,
            beta=None,
            gamma=None)
        fname_alphas3.append(fname_alphas)
        valid_alphas3.append(valid_alphas)
        final_perf_accs_mean3.append(final_perf_accs_mean)
        final_perf_accs_err3.append(final_perf_accs_err)

    color = 'navy'

    for i in range(3):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        plt.errorbar(valid_alphas3[i], final_perf_accs_mean3[i], yerr=np.array(final_perf_accs_err3[i]).T, fmt='o',
                     color=color)

        plt.ylim(0.9, 1.01)
        plt.xlim(0, valid_alphas3[i][-1] + 0.1 * (valid_alphas3[i][-1] - valid_alphas3[i][0]))
        plt.ylabel('Final accuracy', fontsize=14)
        if rule != 'size':
            plt.xlabel(r'$\alpha_{%s}$' % rule, fontsize=14)
        else:
            plt.xlabel(r'$\alpha$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + '_' + rule + '_accs_vs_alphas.png', bbox_inches='tight',
                    transparent=True, dpi=500)
    plt.show()


def ANN_adaptation_pattern():

    dirs = ["results/res_rules=(0,0,2,0,0)_betas=pm_np.linspace(0.1,1,9)_with_neg_betas=False_nets_per_total_alpha=50"]

    fname_alphas3, valid_alphas3, yields3, yield_errs3, ratios3, ratios_errs3, betas3 = [], [], [], [], [], [], []
    for dir in dirs:
        if 'pm' in dir:
            betas1 = eval(
                dir[dir.find('betas=pm_') + len('betas=pm_'):[m.start() for m in re.finditer("\)", dir)][1] + 1])
            betas = np.sort(np.concatenate([betas1, -betas1]))
        else:
            betas = eval(dir[dir.find('betas=') + len('betas='):[m.start() for m in re.finditer("\)", dir)][1] + 1])
        # betas = betas[:-2]
        fname_alphas_eta = []
        valid_alphas_eta = []
        yields_eta = []
        yield_errs_eta = []
        ratios_eta = []
        ratios_errs_eta = []
        for beta in betas:
            fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, _, _, _, _, _ = concept_rule_ratio_for_ANN_a1_to_a2(dir,
                                                                                                                    eta_ratio=None,
                                                                                                                    beta=beta,
                                                                                                                    gamma=None,
                                                                                                                    gamma_in = None,
                                                                                                                    mask='alpha_from_0.1')
            fname_alphas_eta.append(fname_alphas)
            valid_alphas_eta.append(valid_alphas)
            yields_eta.append(yields)
            yield_errs_eta.append(yield_errs)
            ratios_eta.append(ratios)
            ratios_errs_eta.append(ratios_errs)
        fname_alphas3.append(fname_alphas_eta)
        valid_alphas3.append(valid_alphas_eta)
        yields3.append(yields_eta)
        yield_errs3.append(yield_errs_eta)
        ratios3.append(ratios_eta)
        ratios_errs3.append(ratios_errs_eta)
        betas3.append(betas)


    r = ratios3
    a2 = valid_alphas3
    a1 = betas3
    r, a1, a2 = r[0], a1[0], a2[0]

    r_array = np.array(r).T
    be_array = np.array(a1)
    al_array = np.array(a2[0])

    # Now, define the masks for positive and negative β:
    pos_mask = be_array >= 0
    neg_mask = be_array < 0

    X_pos, Y_pos = np.meshgrid(be_array[pos_mask], al_array)
    R_pos = r_array[:, pos_mask]

    X_neg, Y_neg = np.meshgrid(al_array, np.abs(be_array[neg_mask]))

    R_neg = r_array[:, neg_mask].T



    x_fine = np.linspace(X_neg.min(), X_neg.max(), 500)
    y_fine = np.linspace(Y_neg.min(), Y_neg.max(), 500)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    R_fine = griddata((X_neg.ravel(), Y_neg.ravel()), 100. * R_neg.ravel(), (X_fine, Y_fine), method='cubic')

    R_neg *= 100


    # Separate points based on R_neg threshold
    mask_triangles = R_neg < 50
    mask_squares = R_neg >= 50

    # Create figure and axes with constrained layout
    plt.figure(figsize=(6, 6))

    # Scatter plot with triangles for R_neg < 50
    plt.scatter(X_neg[mask_triangles], Y_neg[mask_triangles], c=R_neg[mask_triangles],
                cmap='magma', s=50, edgecolors='k', alpha=0.75, marker='^', vmin=0, vmax=100)

    # Scatter plot with squares for R_neg > 50
    plt.scatter(X_neg[mask_squares], Y_neg[mask_squares], c=R_neg[mask_squares],
                cmap='magma', s=50, edgecolors='k', alpha=0.75, marker='s', vmin=0, vmax=100)

    # Colorbar
    reversed_cmap = mcolors.ListedColormap(plt.cm.magma(np.linspace(1, 0, 256)))

    # Create a ScalarMappable with the reversed colormap
    sm = plt.cm.ScalarMappable(cmap=reversed_cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])

    # Colorbar
    # cbar = plt.colorbar(sm, ax=plt.gca())
    cbar = plt.gcf().colorbar(sm, ax=plt.gca(), fraction=0.045, pad=0.04)
    # Force aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Adjust the right margin to leave space for colorbar
    plt.gcf().subplots_adjust(right=0.8)

    # Add the colorbar

    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([0, 20, 40, 60, 80, 100])
    # cbar.set_label(r"% adapted $\Delta Z$", fontsize=14)
    cbar.set_label(r'Fraction of $\theta$' + '\nchanging sign (%)', fontsize=14)

    # Labels
    plt.xlabel(r'$\alpha_1$', fontsize=16)
    plt.ylabel(r'$\alpha_2$', fontsize=16)

    plt.plot(al_array, 0.34**2/al_array, '-', color='black', lw=2)
    plt.plot(al_array, al_array, '--', color='black', lw=2)

    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # Tick font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    # Save the figure
    plt.savefig('ANNs_a1_to_a2.png', bbox_inches='tight', transparent=True, dpi=500)

    plt.show()


if __name__ == '__main__':
    # Figure 1
    # import Images
    # Images.create_batch()
    # Images.plot_batch('alpha=0.5_rule=(0, 0, 2, 0, 0)_0.png')

    # Figure 2 a
    # fname = 'results/res_rules=(0,0,2,0,0)_just_training=True_measure_optimization=True_total_alphas=_0.5__nets_per_total_alpha=100/alpha=0.500000.pt'
    # plot_just_training(fname)

    # Figure 2 b
    # fname_base = 'results/res_rules=(0,0,2,0,0)_nets_per_total_alpha=100'
    # acc_vs_alpha(fname_base)

    # Figure 2 c-d
    # folder = 'results/res_rules=(0,0,2,0,0)_total_alphas=_0.2,0.8__measure_optimization=True_nets_per_total_alpha=50'
    # two_fixed_points(folder, net=0)
    # two_fixed_points(folder, net=1)

    # Figure 3
    # folder = 'results/res_rules=(0,0,2,0,0)_total_alphas=_0.2,0.8__measure_optimization=True_nets_per_total_alpha=50'
    # plot_two_alphas_together(folder, mask='invert_dzs')

    # Figure 4
    # fname_base = 'results/res_rules=(2,0,0,0,0)_nets_per_total_alpha=100'
    # plot_concept_rule_ratios(fname_base)

    # Figure 5
    # ANN_adaptation_pattern()

    # Figure 6
    # fname_base = 'results/res_rules=(2,0,0,0,0)_betas=np.linspace(0.1,1,9)_nets_per_total_alpha=50'
    # plot_betas(fname_base)

    # Figure 7
    # import curriculum_quivers as cq
    # cq.create_single_quiver_plot(alpha_from=0.5, alpha_to=-0.5, lambda_=0.1)
    # plt.show()
    # cq.create_single_quiver_plot(alpha_from=2, alpha_to=-2, lambda_=0.1)
    # plt.show()

    # Figure 8
    # import curriculum_quivers as cq
    # fig_alpha_tos = cq.plot_different_alpha_tos()
    # plt.show()
    # fig_alpha_ratios = cq.plot_different_alpha_ratios()
    # plt.show()
    # cq.plot_crossing_phase_diagram(lambda_=0.1)
    # plt.show()

    pass
