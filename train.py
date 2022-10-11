import os

import torch
import time
import numpy as np
from tqdm import tqdm
from model import Linear, Conv
from data import iGEM_data
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
prefix = os.getcwd() + '/../result/'
loss_set = []
epoch_loss = []


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X1, y) in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            X1, y = X1.to(device), y.to(device)

            # Compute prediction error
            pred = model(X1)
            loss = loss_fn(pred, y.long())

            loss_set.append(loss.cpu().detach().numpy())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            time.sleep(0.0001)

            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(X1)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    correct = 0
    with torch.no_grad():
        for X1, y in dataloader:
            X1, y = X1.to(device), y.to(device)
            pred = model(X1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test: Avg loss: {test_loss:>8f}, Accuracy: {accuracy:>8f} \n")
    return test_loss


if __name__ == '__main__':
    models = {'Linear': Linear(), 'Conv': Conv()}
    for model_id in models:
        current_model = models[model_id]
        train_dataset = iGEM_data((0, 4000))
        test_dataset = iGEM_data((4000, 5000))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

        print(f'Running {model_id} model......')
        current_model.to(device)
        optimizer = torch.optim.Adam(current_model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        epochs = 32
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            train(train_dataloader, current_model, loss_fn, optimizer, epoch=t)
            np.save(f'{prefix}loss_set/{model_id}/Epoch{t + 1}.npy', loss_set)

            loss_set.clear()

            now = test(test_dataloader, current_model, loss_fn)

            # d_now = now.cpu().detach().numpy()
            d_now = now
            epoch_loss.append(d_now)

        np.save(f'{prefix}loss_set/{model_id}/Epoch_test_loss.npy', epoch_loss)
        torch.save(current_model.state_dict(), f'{prefix}model_weights_{model_id}.pth')
        print(f"Model {model_id} is done!\n")
