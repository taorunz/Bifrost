import time
import sys

import torch

from utils import torch_device

def train(model, train_dl, epochs, loss_fn, optimizer, device=None, val_dl=None, acc_fn=None, cont_print=False):
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        
        batches = 0
        total_batches = len(train_dl)
        
        start_time = time.time()
        
        for batch_dict in train_dl:
            x = batch_dict['image']
            y = batch_dict['digit']
            
            y = y.to(torch.long)   

            x = x.to(torch_device)
            y = y.to(torch_device)
            
            optimizer.zero_grad()
            
            if model.service == "TorchQuantum":
                preds = model(device, x)
            else:
                preds = model(x)

            loss = loss_fn(preds, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batches += 1
            
            cur_time = time.time()
            avg_batch_time = (cur_time-start_time)/batches

            if cont_print:
                print(f"Epoch {epoch + 1} | Loss: {running_loss/batches} | Est. Time Remaining: {avg_batch_time*(total_batches-batches)}", end="\r", file=sys.stderr)
        
        print(f"Epoch {epoch + 1} | Loss: {running_loss/batches} | Time Elapsed: {cur_time-start_time}", file=sys.stderr)
        losses.append(running_loss/batches)

        if val_dl and acc_fn:
            val_loss, val_acc = test(model, eval_dl=val_dl, acc_fn=acc_fn, loss_fn=loss_fn, device=device)
            print(f"Validation | Loss: {val_loss}, Accuracy: {val_acc}", file=sys.stderr)

    return losses

def test(model, eval_dl, acc_fn, loss_fn, device=None):
    preds = []
    labels = []

    with torch.no_grad():
        for _, batch_dict in enumerate(eval_dl):
            x = batch_dict['image']
            y = batch_dict['digit']

            x = x.to(torch_device)
            y = y.to(torch_device)

            if model.service == "TorchQuantum":
                batch_preds = model(device, x)
            else:
                batch_preds = model(x)

            preds.append(batch_preds)
            labels.append(y)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
    
        loss = loss_fn(preds, labels).item()
        accuracy = acc_fn(preds, labels)

    return loss, accuracy