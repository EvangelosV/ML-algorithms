import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model, train_loader, dev_loader, optimizer, criterion, num_epochs, device):
    model.to(device)
    train_losses = []
    dev_losses = []
    best_dev_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * texts.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for texts, labels in dev_loader:
                texts = texts.to(device)
                labels = labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                dev_loss += loss.item() * texts.size(0)
        dev_loss /= len(dev_loader.dataset)
        dev_losses.append(dev_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    return model, train_losses, dev_losses, best_epoch

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return all_labels, all_preds

def plot_loss_curves(train_losses, dev_losses):
    epochs = range(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, dev_losses, marker='s', label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.tight_layout()
    filename = "loss_curves.png"
    plt.savefig(filename)
    plt.close()
    #plt.show()
