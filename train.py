from tqdm import tqdm

import config
import torch
import metrics
import os
import utils
import numpy as np

# Training loop
def train(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    true_labels = []
    pred_labels = []
    for images, labels in tqdm(train_loader):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        # print("labels shape",labels.shape,"pred shape",predicted.shape)
        pred_labels.extend(predicted.cpu().numpy())
        
    true_labels = torch.tensor(true_labels)
    pred_labels = torch.tensor(pred_labels)
    acc = metrics.accuracy(true_labels, pred_labels)
    return running_loss / len(train_loader.dataset) , acc

# Testing loop
def test(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / len(test_loader.dataset)
    true_labels = torch.tensor(true_labels)
    pred_labels = torch.tensor(pred_labels)
    acc = metrics.accuracy(true_labels, pred_labels)
    macro_f1 = metrics.f1_score(true_labels, pred_labels)

    return avg_loss, acc, macro_f1



def fit(model, criterion, optimizer, train_loader, val_loader,test_loader,num_epochs, log_file, save_dir):
    train_losses = np.zeros(num_epochs)
    train_accuricies = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    val_f1_scores = np.zeros(num_epochs)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss , train_acc = train(model, criterion, optimizer, train_loader)
        val_loss, val_acc, macro_f1 = test(model, criterion, val_loader)
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss
        val_accuracies[epoch] = val_acc
        val_f1_scores[epoch] = macro_f1
        utils.log_training_process(epoch, train_loss, val_loss, val_acc, macro_f1)
        print(f"Epoch {epoch+1} | train_loss: {train_loss:.3f} | train_acc: {train_acc:.3f} | val_loss: {val_loss:.3f} | val_acc: {val_acc:.3f} | f1_score: {macro_f1:.3f}")
        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save({
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "train_loss":train_loss,
                "val_loss":val_loss,
                "val_accuracy":val_acc,
                "val_f1_score":macro_f1
            },config.LOGFILE)

    test_loss, test_acc, test_macro_f1 = test(model,criterion,test_loader)
    print(f"test_loss: {test_loss:.3f} | test_acc: {test_acc:.3f} | f1_score: {test_macro_f1:.3f}")

    print("Training completed!")
    return train_losses,train_accuricies,val_losses,val_accuracies,val_f1_scores
