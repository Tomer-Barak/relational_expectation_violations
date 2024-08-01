import glob
import ast
import inspect
import re
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


def plot_RTs(fname_base):
    rules = ['(2,0,0,0,0)', '(0,0,2,0,0)', '(0,0,0,0,2)']

    dirs = [fname_base[:fname_base.find('rules=') + len('rules=')] + rule + fname_base[fname_base.find('rules=') + len(
        'rules=') + len(rule):] for rule in rules]

    fname_base = (fname_base[:fname_base.find('rules=') - 1] +
                  fname_base[fname_base.find('rules=') + len('rules=') + len(rules[0]):])

    fname_alphas3, valid_alphas3, yields3, yield_errs3, RT3, RT_errs3, RTlist3 = [], [], [], [], [], [], []
    for dir in dirs:
        fname_alphas, valid_alphas, yields, yield_errs, _, _, RT, RT_errs, RTlist, _, _ = concept_rule_ratio(dir,
                                                                                                             gamma=None,
                                                                                                             beta=None,
                                                                                                             eta_ratio=None)
        fname_alphas3.append(fname_alphas)
        valid_alphas3.append(valid_alphas)
        yields3.append(yields)
        yield_errs3.append(yield_errs)
        RT3.append(RT)
        RT_errs3.append(RT_errs)
        RTlist3.append(RTlist)

    all_RTs = np.vstack([alpha for rule in RTlist3 for alpha in rule])

    t_statistic, p_value = st.ttest_rel(all_RTs[:, 1], all_RTs[:, 0], alternative='greater')

    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(8.27, 2.5)
    for i in range(3):
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        plt.sca(ax[i])
        RT = np.array(RT3[i])
        RT_errs = np.array(RT_errs3[i])
        plt.bar(np.array(valid_alphas3[i]) - 0.01, 2 * RT[:, 0], yerr=2 * RT_errs[:, 0], width=0.02, label='RT 1')
        plt.bar(np.array(valid_alphas3[i]) + 0.01, 2 * RT[:, 1], yerr=2 * RT_errs[:, 1], width=0.02, label='RT 2')
        ylims = plt.gca().get_ylim()
        plt.ylim(0, ylims[1])
        plt.xlim(0, fname_alphas3[i][-1] + 0.05 * (fname_alphas3[i][-1] - fname_alphas3[i][0]))
        plt.ylabel('Examples to success', fontsize=12)
        plt.xlabel(r'$|\alpha_{%s}|$' % rule, fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + '_RTs.png', bbox_inches='tight', transparent=True, dpi=500)

    plt.show()


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

        T_s_mean = np.mean(T_s, axis=0)
        T_s_error = 1.96 * st.sem(T_s, axis=0)

        dz_means_mean = np.mean(dz_means, axis=0)
        dz_means_error = 1.96 * st.sem(dz_means, axis=0)

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


def plot_two_alphas_together(mask=None):
    folder = 'results/res_rules=(0,0,0,0,2)_total_alphas=_0.2,0.8__measure_optimization=True_nets_per_total_alpha=50'
    fname1 = folder + '/alpha=0.800000.pt'
    fname2 = folder + '/alpha=0.200000.pt'

    rules = folder[folder.find('rules=') + len('rules='):folder.find('rules=') + len('rules=') + len('(0,0,0,0,0)')]
    rules = ast.literal_eval(rules)
    possible_rules = ['color', '', 'size', '', 'number']
    rule = possible_rules[rules.index(2)]

    measures1 = torch.load(fname1)
    HP1, Alphas1, dz_means_mean1, dz_means_error1, T_s_mean1, T_s_error1, perf_accs_mean1, perf_accs_error1, _ = average_measures(
        measures1, mask)

    measures2 = torch.load(fname2)
    HP2, Alphas2, dz_means_mean2, dz_means_error2, T_s_mean2, T_s_error2, perf_accs_mean2, perf_accs_error2, _ = average_measures(
        measures2, mask)

    epochs = np.arange(0, HP1['batches_per_alpha'] * HP1['alpha_steps'] + 1) * HP1['batch']
    middle_epoch_ind = HP1['batches_per_alpha'] + 1

    # fig, ax = plt.subplots(1, 4)
    # fig.set_size_inches(8.27, 2.3)
    color1 = 'navy'
    color2 = 'crimson'

    # plt.sca(ax[0])
    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.plot(epochs, Alphas1, lw=2, color=color1)
    plt.plot(epochs, Alphas2, lw=2, color=color2)
    plt.ylabel(r'$\alpha_{%s}$' % rule, fontsize=14)  # , rotation=0)
    plt.xlabel('Example pairs', fontsize=12)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    yticks = np.sort(np.concatenate((np.unique(Alphas1), np.unique(Alphas2), np.array([0.]))))
    plt.yticks(yticks, fontsize=12)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1] + '_two_alphas_alphas.png', transparent=True, bbox_inches='tight', dpi=600)

    # plt.sca(ax[1])
    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0.5, color='black', linestyle='--', lw=1)
    plt.plot(epochs, perf_accs_mean1, lw=2, color=color1)
    plt.fill_between(epochs, perf_accs_mean1 - perf_accs_error1, perf_accs_mean1 + perf_accs_error1, alpha=0.3,
                     color=color1)
    plt.plot(epochs, perf_accs_mean2, lw=2, color=color2)
    plt.fill_between(epochs, perf_accs_mean2 - perf_accs_error2, perf_accs_mean2 + perf_accs_error2, alpha=0.3,
                     color=color2)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Example pairs', fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1]  + '_two_alphas_accs.png', transparent=True, bbox_inches='tight', dpi=600)

    # plt.sca(ax[2])
    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.plot(epochs, dz_means_mean1, lw=2, color=color1)
    plt.fill_between(epochs, dz_means_mean1 - dz_means_error1, dz_means_mean1 + dz_means_error1, alpha=0.3,
                     color=color1)
    plt.plot(epochs, dz_means_mean2, lw=2, color=color2)
    plt.fill_between(epochs, dz_means_mean2 - dz_means_error2, dz_means_mean2 + dz_means_error2, alpha=0.3,
                     color=color2)
    plt.ylabel(r'$\Delta z$', fontsize=14, rotation=0, labelpad=10)
    plt.xlabel('Example pairs', fontsize=12)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1]  + '_two_alphas_dz.png', transparent=True, bbox_inches='tight', dpi=600)

    plt.figure(figsize=(4, 3))
    plt.axvline(x=epochs[middle_epoch_ind], color='black', linestyle='--', lw=1)
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.plot(epochs, T_s_mean1, lw=2, color=color1)
    plt.fill_between(epochs, T_s_mean1 - T_s_error1, T_s_mean1 + T_s_error1, alpha=0.3,
                     color=color1)
    plt.plot(epochs, T_s_mean2, lw=2, color=color2)
    plt.fill_between(epochs, T_s_mean2 - T_s_error2, T_s_mean2 + T_s_error2, alpha=0.3,
                     color=color2)
    plt.ylabel(r'$\theta$', fontsize=14, rotation=0, labelpad=10)
    plt.xlabel('Example pairs', fontsize=12)
    plt.xticks([0, epochs[middle_epoch_ind - 1], epochs[-1]], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(folder.split('/')[1]  + '_two_alphas_theta.png', transparent=True, bbox_inches='tight', dpi=600)

    plt.show()


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
    for fname in fnames:
        all_measures = torch.load(fname)

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
                    print(fname)
                    print(40 * 'X' + ' Invalid network ' + 40 * 'X')
                    print(dz_s)
                    print(T_params)
                    print(perf_accs)
                    # exit()

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

            if len(all_measures) > 1:

                ratios_errs.append(
                    1.96 * np.sqrt((ratios[-1] * (1 - ratios[-1])) / (concept_changes + rule_changes - 1)))

            else:
                ratios_errs.append(0)
            valid_alphas.append(HP['total_alpha'])

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

    color = 'navy'

    for i in range(3):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        # plt.sca(ax[i])
        fit_color = 'crimson'
        guess = (1, 0.5)
        popt, pcov = curve_fit(sigmoid, valid_alphas3[i], ratios3[i], guess)
        plt.plot(valid_alphas3[i], 100 * sigmoid(valid_alphas3[i], *popt), color=fit_color)
        plt.axvline(popt[1], 0, 0.97, color='gray', linestyle='--', lw=1)
        plt.errorbar(valid_alphas3[i], 100 * np.array(ratios3[i]), yerr=100 * np.array(ratios_errs3[i]).T, fmt='o',
                     color=color)

        plt.text(popt[1] + 0.02, -0.2, r'$\bar\alpha$', fontsize=14, ha='left', va='bottom',
                 transform=plt.gca().transData)
        print(rule, popt[1], 1.96 * np.sqrt(np.diag(pcov))[1])

        plt.ylim(-0.1 * 100, 1.1 * 100)
        plt.xlim(0, valid_alphas3[i][-1] + 0.1 * (valid_alphas3[i][-1] - valid_alphas3[i][0]))
        # plt.xscale('log')
        plt.ylabel(r'% adapted $\Delta Z$', fontsize=14)
        plt.xlabel(r'$|\alpha_{%s}|$' % rule, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1]  + '_' + rule + '_concept_ratios.png', bbox_inches='tight', transparent=True, dpi=500)

    plt.show()


def plot_eta_ratios(fname_base):
    rules = ['(2,0,0,0,0)', '(0,0,2,0,0)', '(0,0,0,0,2)']

    dirs = [fname_base[:fname_base.find('rules=') + len('rules=')] + rule + fname_base[fname_base.find('rules=') + len(
        'rules=') + len(rule):] for rule in rules]

    fname_base = (fname_base[:fname_base.find('rules=') - 1] +
                  fname_base[fname_base.find('rules=') + len('rules=') + len(rules[0]):])

    fname_alphas3, valid_alphas3, yields3, yield_errs3, ratios3, ratios_errs3, eta_ratios3 = [], [], [], [], [], [], []
    for dir in dirs:
        eta_ratios = eval(dir[dir.find('eta_ratios=') + len('eta_ratios='):dir.find('eta_ratios=') + len(
            'eta_ratios=') + dir[dir.find(
            'eta_ratios=') + len('eta_ratios='):].find(')') + 1])
        eta_ratios = eta_ratios[:-2]
        fname_alphas_eta = []
        valid_alphas_eta = []
        yields_eta = []
        yield_errs_eta = []
        ratios_eta = []
        ratios_errs_eta = []
        for eta_ratio in eta_ratios:
            fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, _, _, _, _, _ = concept_rule_ratio(dir,
                                                                                                                    eta_ratio,
                                                                                                                    beta=None,
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
        eta_ratios3.append(eta_ratios)

    color = 'navy'
    fit_color = 'crimson'

    alpha_stars3 = []
    alpha_stars_errs3 = []
    valid_etas = []
    for i in range(len(fname_alphas3)):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        alpha_stars_eta = []
        alpha_stars_errs_eta = []
        valid_etas_alpha = []
        for j in range(len(fname_alphas3[0])):
            plt.errorbar(valid_alphas3[i][j], 100 * np.array(ratios3[i][j]), yerr=100 * np.array(ratios_errs3[i][j]),
                         color=color,
                         alpha=(j + 0.3) / (len(fname_alphas3[0]) + 0.3))
            guess = (1, 0.3)
            popt, pcov = curve_fit(sigmoid, valid_alphas3[i][j], ratios3[i][j], guess)
            perr = np.sqrt(np.diag(pcov))
            plt.plot(valid_alphas3[i][j], 100 * sigmoid(valid_alphas3[i][j], *popt), color=fit_color,
                     alpha=(j + 0.3) / (len(fname_alphas3[0]) + 0.3))
            alpha_stars_eta.append(popt[1])
            alpha_stars_errs_eta.append(perr[1] * 1.96)
            valid_etas_alpha.append(eta_ratios3[i][j])

        valid_etas.append(valid_etas_alpha)

        alpha_stars3.append(alpha_stars_eta)
        alpha_stars_errs3.append(alpha_stars_errs_eta)
        plt.ylim(-0.1 * 100, 1.1 * 100)

        plt.ylabel(r'% adapted $\Delta Z$', fontsize=14)
        plt.xlabel(r'$|\alpha_{%s}|$' % rule, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + f'_{rule}_concept_ratios.png', bbox_inches='tight', transparent=True, dpi=500)

    for i in range(len(fname_alphas3)):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]

        plt.errorbar(1 / np.sqrt(eta_ratios3[i]), (alpha_stars3[i]), yerr=alpha_stars_errs3[i], color=color, fmt='o')

        plt.ylabel(r'$|\bar{\alpha}_{%s}|$' % rule, fontsize=14)
        plt.xlabel(r'$\sqrt{\eta_\theta/\eta_w}$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + f'_{rule}_eta_ratios.png', bbox_inches='tight', transparent=True, dpi=500)
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
    fit_color = 'crimson'

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
        plt.ylabel(r'$\bar{\alpha}_{%s}$' % rule, fontsize=14)
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


def plot_gammas(fname_base):
    rules = ['(2,0,0,0,0)', '(0,0,2,0,0)', '(0,0,0,0,2)']

    dirs = [fname_base[:fname_base.find('rules=') + len('rules=')] + rule + fname_base[fname_base.find('rules=') + len(
        'rules=') + len(rule):] for rule in rules]

    fname_base = (fname_base[:fname_base.find('rules=') - 1] +
                  fname_base[fname_base.find('rules=') + len('rules=') + len(rules[0]):])

    fname_alphas3, valid_alphas3, yields3, yield_errs3, ratios3, ratios_errs3, gammas3 = [], [], [], [], [], [], []
    for dir in dirs:
        gammas = dir[dir.find('gammas=') + len('gammas='):dir.find('gammas=') + len('gammas=') + dir[
                                                                                                 dir.find(
                                                                                                     'gammas=') + len(
                                                                                                     'gammas='):].find(
            ')') + 1]
        if '_' in gammas:
            gammas = gammas.replace('_', '/')
        gammas = eval(gammas)
        fname_alphas_eta = []
        valid_alphas_eta = []
        yields_eta = []
        yield_errs_eta = []
        ratios_eta = []
        ratios_errs_eta = []
        for gamma in gammas:
            fname_alphas, valid_alphas, yields, yield_errs, ratios, ratios_errs, _, _, _, _, _ = concept_rule_ratio(dir,
                                                                                                                    eta_ratio=None,
                                                                                                                    beta=None,
                                                                                                                    gamma=gamma)
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
        gammas3.append(gammas)

    color = 'navy'
    fit_color = 'crimson'

    alpha_stars3 = []
    alpha_stars_errs3 = []
    valid_gammas3 = []
    for i in range(len(fname_alphas3)):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        alpha_stars_eta = []
        alpha_stars_errs_eta = []
        valid_gammas_eta = []

        plt_alphas = np.array(gammas3[i])
        plt_alphas /= np.max(plt_alphas)
        plt_alphas -= np.min(plt_alphas)

        for j in range(len(fname_alphas3[0])):
            plt.errorbar(valid_alphas3[i][j], 100 * np.array(ratios3[i][j]), yerr=100 * np.array(ratios_errs3[i][j]),
                         color=color,
                         alpha=(plt_alphas[j] + 0.3) / (1 + 0.3))
            guess = (1, 0.5)
            try:
                popt, pcov = curve_fit(sigmoid, valid_alphas3[i][j], ratios3[i][j], guess)
                perr = np.sqrt(np.diag(pcov))
                plt.plot(valid_alphas3[i][j], 100 * sigmoid(valid_alphas3[i][j], *popt), color=fit_color,
                         alpha=(plt_alphas[j] + 0.3) / (1 + 0.3))
                if popt[1] >= 0:
                    alpha_stars_eta.append(popt[1])
                    alpha_stars_errs_eta.append(perr[1] * 1.96)
                    valid_gammas_eta.append(gammas3[i][j])
            except:
                pass
        valid_gammas3.append(valid_gammas_eta)

        alpha_stars3.append(alpha_stars_eta)
        alpha_stars_errs3.append(alpha_stars_errs_eta)
        plt.ylim(-0.1 * 100, 1.1 * 100)

        plt.ylabel(r'% adapted $\Delta Z$', fontsize=14)
        plt.xlabel(r'$|\alpha_{%s}|$' % rule, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + f'_{rule}_concept_ratios.png', bbox_inches='tight', transparent=True, dpi=500)

    for i in range(len(fname_alphas3)):
        plt.figure(figsize=(4, 3))
        actual_rules = ast.literal_eval(rules[i])
        possible_rules = ['color', '', 'size', '', 'number']
        rule = possible_rules[actual_rules.index(2)]
        plt.errorbar(1 / np.array(valid_gammas3[i]), (alpha_stars3[i]), yerr=np.array(alpha_stars_errs3[i]),
                     color=color, fmt='o')

        plt.ylabel(r'$|\bar{\alpha}_{%s}|$' % rule, fontsize=14)
        plt.xlabel(r'$1/\gamma$', fontsize=14)

        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + f'_{rule}_gammas.png', bbox_inches='tight', transparent=True, dpi=500)
    plt.show()


def plot_just_training(fname):
    measures1 = torch.load(fname)
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
        plt.ylabel('Final accuracy', fontsize=12)
        plt.xlabel(r'$\alpha_{%s}$' % rule, fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname_base.split('/')[1] + '_' + rule + '_accs_vs_alphas.png', bbox_inches='tight',
                    transparent=True, dpi=500)
    plt.show()


if __name__ == '__main__':
    # The functions below reconstruct the paper Figures related to the ANN.
    # To run these functions, you need to first obtain the results of the ANN using main.py. See instructions in GitHub.
    # The Figures of the simplified model are plotted within their code files (see the folder "\linear")

    # Figure 2 left
    # fname = 'results/res_rules=(0,0,2,0,0)_just_training=True_measure_optimization=True_total_alphas=_0.5__nets_per_total_alpha=100/alpha=0.500000.pt'
    # plot_just_training(fname)

    # Figure 2 right
    # fname_base = 'results/res_rules=(2,0,0,0,0)'
    # acc_vs_alpha(fname_base)

    # Figures 4 and 5
    # plot_two_alphas_together(mask='invert_dzs')

    # Figure 6
    # fname_base = 'results/res_rules=(2,0,0,0,0)_nets_per_total_alpha=100'
    # plot_concept_rule_ratios(fname_base)

    # Figure S5
    # fname_base = 'results/res_rules=(0,0,0,0,2)_only_for_RT=True_measure_optimization=True_nets_per_total_alpha=50'
    # plot_RTs(fname_base)

    # Figure 8
    # fname_base = 'results/res_rules=(2,0,0,0,0)_gammas=1_np.linspace(0.25,4,9)_nets_per_total_alpha=50'
    # plot_gammas(fname_base)

    # Figure 9
    # fname_base = 'results/res_rules=(2,0,0,0,0)_eta_ratios=np.logspace(-2,2,9,base=2)_nets_per_total_alpha=50'
    # plot_eta_ratios(fname_base)

    # Figure 12 right
    # fname_base = 'results/res_rules=(2,0,0,0,0)_betas=np.linspace(0.1,1,9)_nets_per_total_alpha=50'
    # plot_betas(fname_base)

    pass
