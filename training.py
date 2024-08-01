import torch
import numpy as np
import itertools
import Images as imgs


def loss_fn(e1, e_t_0):
    loss = 0
    for i in range(len(e1)):
        loss += torch.norm(e_t_0[i] - e1[i]) ** 2
    loss /= len(e_t_0)
    return loss


def optimize(HP, Z, T, Z_test_images, perf_test_images):
    if HP['eta_ratio'] == 1.:
        optim = torch.optim.SGD(filter(lambda h: h.requires_grad, itertools.chain(Z.parameters(), T.parameters())),
                                lr=HP['lr'], momentum=HP['momentum'])
    else:
        optimZ = torch.optim.SGD(Z.parameters(), lr=HP['eta_ratio'] * HP['lr'], momentum=HP['momentum'])
        optimT = torch.optim.SGD(T.parameters(), lr=HP['lr'], momentum=HP['momentum'])

    dz_s = []
    T_params = []
    perf_accs = []
    Mean_losses = []
    RT = []

    RT_measure_flag = True

    for epoch in range(HP['batches_per_alpha']):
        images = imgs.create_batch(HP)
        if HP['GPU']:
            images = images.cuda()

        if HP['measure_optimization'] and RT_measure_flag:
            dz, T_param, perf_acc = evaluate(HP, Z, T, Z_test_images, perf_test_images)
            dz_s.append(dz)
            T_params.append(T_param)
            perf_accs.append(perf_acc)
            if HP['only_for_RT'] and perf_acc > 0.8:
                RT_measure_flag = False
                RT.append(epoch)
                break

        losses = []
        for step in range(HP['steps_per_batch']):
            if HP['eta_ratio'] == 1.:
                optim.zero_grad()
            else:
                optimZ.zero_grad()
                optimT.zero_grad()
            z_0 = Z(images[:, 0, :])
            t_z_0 = T(z_0)
            z_1 = Z(images[:, 1, :])
            loss = loss_fn(z_1, t_z_0)

            if HP['circle_regularization']:
                reg_term1 = T.const1.weight.norm() ** 2
                reg_term2 = torch.mean(z_1 - z_0) ** 2
                radius = 0.1
                loss += HP['circle_regularization'] * (reg_term1 + reg_term2 - radius) ** 2

            losses.append(loss.item())
            loss.backward()
            if HP['eta_ratio'] == 1.:
                optim.step()
            else:
                optimZ.step()
                optimT.step()

    if not HP['only_for_RT'] or RT_measure_flag:
        dz, T_param, perf_acc = evaluate(HP, Z, T, Z_test_images, perf_test_images)
        dz_s.append(dz)
        T_params.append(T_param)
        perf_accs.append(perf_acc)
        Mean_losses.append(np.mean(losses))

    measures = {'Mean_losses': Mean_losses, 'dz_s': dz_s, 'T_params': T_params, 'perf_accs': perf_accs, 'RT': RT}
    return measures


def evaluate(HP, Z, T, Z_test_images, perf_test_images):
    dz = Z(Z_test_images[:, 1, :]) - Z(Z_test_images[:, 0, :])
    dz_mean = torch.mean(dz, dim=0).item()
    dz_err = 2 * torch.std(dz, dim=0).item() / np.sqrt(len(dz) - 1)
    epsilon = 1e-10
    if dz_err > epsilon:
        dz_mean = np.round(dz_mean, -int(np.floor(np.log10(abs(dz_err)))))
    else:
        dz_mean = np.round(dz_mean, -int(np.floor(np.log10(epsilon))))

    T_param = T.const1.weight.data.item()

    z_0 = Z(perf_test_images[:, 0, :])
    t_z_0 = T(z_0)
    z_1 = Z(perf_test_images[:, 1, :])
    t_z_1 = T(z_1)
    perf_probs = torch.softmax(-torch.cat((torch.square(t_z_0 - z_1), torch.square(t_z_1 - z_0)), dim=1), dim=1)
    perf_accuracy = (sum(perf_probs[:, 0] > 0.5) / len(perf_probs)).item()

    return dz_mean, T_param, perf_accuracy
