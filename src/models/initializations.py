import torch
import numpy as np


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def first_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def second_last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            assert m.weight.shape == (num_output, num_output)
            m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
            m.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
            m.weight.data /= 30
            m.bias.data /= 30


def last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (1, num_input)
            assert m.bias.shape == (1,)
            # m.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
            m.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
            m.bias.data = torch.zeros(1) + num_input


def first_layer_mfgi_init(m, periods=(1, 30), portion_per_period=(0.25, 0.75)):
    periods = list(periods)  # Number of periods of sine the values of each section of the output vector should hit
    portion_per_period = np.array(list(portion_per_period))
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            num_output = m.weight.size(0)
            num_per_period = (portion_per_period * num_output).astype(int)  # Number of values per section/period
            assert len(periods) == len(num_per_period)
            assert sum(num_per_period) == num_output
            weights = []
            for i in range(0, len(periods)):
                period = periods[i]
                num = num_per_period[i]
                scale = 30 / period
                weights.append(torch.zeros(num, num_input).uniform_(-np.sqrt(3 / num_input) / scale,
                                                                    np.sqrt(3 / num_input) / scale))
            W0_new = torch.cat(weights, axis=0)
            m.weight.data = W0_new


def second_layer_mfgi_init(m, portion_per_period=(0.25, 0.75)):
    portion_per_period = np.array(list(portion_per_period))
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (num_input, num_input)
            num_per_period = (portion_per_period * num_input).astype(int)  # Number of values per section/period
            k = num_per_period[0]  # the portion that only hits the first period
            # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
            W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input),
                                                                np.sqrt(3 / num_input) / 30) * 0.0005
            W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
            W1_new[:k, :k] = W1_new_1
            m.weight.data = W1_new


def geom_relu_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            out_dims = m.out_features

            m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
            m.bias.data = torch.zeros_like(m.bias.data)


def geom_relu_last_layers_init(m):
    radius_init = 1
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
            m.bias.data = torch.Tensor([-radius_init])
