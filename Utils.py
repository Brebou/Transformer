import torch

def save_model(model, path,name):
    torch.save(model.state_dict(), path + name)

def load_model(model, path,name):
    model.load_state_dict(torch.load(path + name))
    model.eval()
    return model



