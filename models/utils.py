import torch


def save_checkpoint(checkpoint_path, **components):
    checkpoint_dict = dict()
    for name, component in components.items():
        if hasattr(component, "state_dict"):
            checkpoint_dict[name] = component.state_dict()
        else:
            checkpoint_dict[name] = component
    torch.save(checkpoint_dict, checkpoint_path)
