import torch


def root_mse(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        score = 0
        score += torch.sqrt(((output-target).pow(2)).mean())
    return score


def legacy_mse(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        score = 0
        score += ((output-target).pow(2)).mean()
    return score
