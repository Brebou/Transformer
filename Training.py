import torch
import time


def train_model(model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    padding_fr,
    padding_en,
    embedding_size = 512,
    num_epochs = 10,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size = 64,
    ):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    time_spent = 0

    model.to(device)

    for epoch in range(num_epochs):
        time_start = time.time()
        model.train()
        train_loss = 0
        accuracy = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            mask = (x == padding_en).unsqueeze(1)  # (batch, 1, seq_len)
            mask = mask.float().masked_fill(mask == 1, float('-inf'))
            non_padding_fr = (y[:, :-1] != padding_fr)
        
            output = model(x, y[:, :-1], mask_input=mask)  
            loss = criterion(output.reshape(-1, output.size(-1)), y[:, 1:].reshape(-1))  
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = output.argmax(dim=-1)  
            correct = (preds == y[:, 1:]).float() * non_padding_fr.float()
            accuracy += correct.sum().item() / non_padding_fr.sum().item()
            if (i+1) % 1 == 0:
                print(f'  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        train_loss /= len(train_loader)
        accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_acc.append(accuracy)
        time_end = time.time()
        time_spent += time_end - time_start
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {accuracy:.4f}, Time: {time_end - time_start:.2f}s')
        model.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                mask = (x == padding_en).unsqueeze(1)  # (batch, 1, seq_len)
                mask = mask.float().masked_fill(mask == 1, float('-inf'))
                non_padding_fr = (y[:, :-1] != padding_fr)
                output = model(x, y[:, :-1], mask_input=mask)  
                loss = criterion(output.reshape(-1, output.size(-1)), y[:, 1:].reshape(-1))  
                
                test_loss += loss.item()

                preds = output.argmax(dim=-1)  
                correct = (preds == y[:, 1:]).float() * non_padding_fr.float()
                accuracy += correct.sum().item() / non_padding_fr.sum().item()
        test_loss /= len(test_loader)
        accuracy /= len(test_loader)
        test_losses.append(test_loss)
        test_acc.append(accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}')

    print(f'Total Training Time: {time_spent:.2f}s')
    return train_losses, test_losses, train_acc, test_acc
