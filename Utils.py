import torch
from torch.utils.data import DataLoader

def save_model(model, path,name):
    torch.save(model.state_dict(), path + name)

def load_model(model, path,name):
    model.load_state_dict(torch.load(path + name))
    model.eval()
    return model

def save_dataset(dataset, path,name):
    # Save the Dataloader object
    torch.save(dataset, path + name)

def load_dataset(path,name):
    # Load the Dataloader object
    dataset = torch.load(path + name)
    return dataset

