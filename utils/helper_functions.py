import sys
sys.path.append("/Users/leo/Desktop/Thesis/utils/")

import time
import shutil
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import random

#setting the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def delete_ds_store(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename == '.DS_Store':
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)

#set the device
def set_device():
    try: 
        device = "mps"
    except:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.device(device)

    return device

def load_model(model, path):
    # Load the state dict
    state_dict = torch.load(path)

    # Load the model weights
    model.load_state_dict(state_dict)

    print(f"Model loaded from {path}")
    return model

def copy_n_folders_with_most_files(src_folder, dest_folder, n):
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]
    subfolder_file_counts = [(subfolder, len(os.listdir(subfolder))) for subfolder in subfolders]
    subfolder_file_counts.sort(key=lambda x: x[1], reverse=True)
    top_n_subfolders = [subfolder for subfolder, file_count in subfolder_file_counts[:n]]
    for subfolder in top_n_subfolders:
        subfolder_name = os.path.basename(subfolder)
        dest = os.path.join(dest_folder, subfolder_name)
        shutil.copytree(subfolder, dest)

def _weights_init(m):
    torch.manual_seed(42)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        #m.bias.data.zero_()

def mean_std_finder(data_path):
    
    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    image_data = torchvision.datasets.ImageFolder(
        root=data_path, transform=transform_img
    )

    image_data_loader = DataLoader(
        image_data, 
        batch_size=len(image_data), 
        shuffle=False, 
        num_workers=0
    )

    def mean_std(loader):
        images, lebels = next(iter(loader))
        # shape of images = [b,c,w,h]
        mean, std = images.mean([0,2,3]), images.std([0,2,3])
        return mean, std
    mean, std = mean_std(image_data_loader)
    print("mean and std: \n", mean, std)

    return mean, std

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def epoch_step_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins, elapsed_secs = divmod(elapsed_time, 60)
    elapsed_secs = int(elapsed_secs)
    elapsed_millisecs = int((elapsed_time - int(elapsed_time)) * 1000)
    return elapsed_mins, elapsed_secs, elapsed_millisecs

# Function to calculate the accuracy of the model
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True) #get the index of the max log-probability
    correct = top_pred.eq(y.view_as(top_pred)).sum() #get the number of correct predictions
    acc = correct.float() / y.shape[0] #calculate the accuracy
    return acc

#train the model
def train(model, train_loader, val_loader, criterion, optimizer, hyper_params, verbose = 0, experiment = False):

    #log hyperparameters
    experiment.log_parameters({key: val for key, val in hyper_params.items() if key != "model"}) if experiment else None

    #send model to device
    model.to(hyper_params["device"])

    #initialise the best validation accuracy
    best_val_acc = 0.0

    #set model to training mode
    model.train()
    #initialise lists to store metrics
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    #loop through epochs
    for epoch in range(hyper_params['epochs']):
        #initialise variables to store metrics
        running_loss = 0.0
        running_corrects = 0

        start_time = time.time() # start time of the epoch
        #loop through batches
        step = 0
        for inputs, labels in tqdm(train_loader, disable= True if verbose == 0 else False):
            #send inputs and labels to device
            inputs = inputs.to(hyper_params["device"])
            labels = labels.to(hyper_params["device"])
            #set gradients to zero
            optimizer.zero_grad()
            #forward pass
            outputs = model(inputs).to(hyper_params["device"])
            #calculate loss
            loss = criterion(outputs, labels)
            #backpropagate
            loss.backward()
            #update weights
            optimizer.step()
            #calculate metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            step += 1

        #calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        #append metrics to lists
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        #validate model
        val_loss, vall_acc = validate(model, val_loader, criterion, hyper_params, verbose = verbose)

        #append metrics to lists
        val_losses.append(val_loss)
        val_acc.append(vall_acc)

        end_time = time.time() # end time of the epoch
        epoch_mins, epoch_secs, epoch_ms = epoch_step_time(start_time, end_time)

        #log metrics to comet_ml
        if experiment:
            experiment.log_metric("train_loss", epoch_loss, step=epoch)
            experiment.log_metric("train_acc", epoch_acc, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)
            experiment.log_metric("val_acc", vall_acc, step=epoch)

        if verbose > 1:
            # Print the statistics of the epoch
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc *100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {vall_acc*100:.2f}%')

        #save the model if it has the best validation accuracy
        if vall_acc > best_val_acc:
            best_val_acc = vall_acc
            torch.save(model.state_dict(), hyper_params["model_save_path"])
            if verbose > 0:
                print(f"Best Accuracy Achieved: {best_val_acc*100:.2f}% on epoch {epoch+1:02}")

    #return metrics
    return train_losses, train_acc, val_losses, val_acc

#validate the model
def validate(model, val_loader, criterion, hyper_params, verbose):
    #set model to evaluation mode
    model.eval()
    #initialise variables to store metrics
    running_loss = 0.0
    running_corrects = 0
    #loop through batches
    for inputs, labels in tqdm(val_loader, disable = True if verbose == 0 else False):
        #send inputs and labels to device
        inputs = inputs.to(hyper_params["device"])
        labels = labels.to(hyper_params["device"])
        #forward pass
        outputs = model(inputs)
        #calculate loss
        loss = criterion(outputs, labels)
        #calculate metrics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    #calculate epoch loss and accuracy
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.float() / len(val_loader.dataset)
    #return metrics
    return epoch_loss, epoch_acc

#test the model
def test(model, test_loader, criterion, hyper_params, experiment = False):
    start_time = time.time() # start time of the epoch

    #load the best model
    model.load_state_dict(torch.load(hyper_params['model_save_path']))
    #set model to evaluation mode
    model.eval()
    #initialise variables to store metrics
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    #loop through batches
    for inputs, labels in test_loader:
        #send inputs and labels to device
        inputs = inputs.to(hyper_params["device"])
        labels = labels.to(hyper_params["device"])
        #model to device
        model = model.to(hyper_params["device"])
        #forward pass
        outputs = model(inputs)
        #calculate loss
        loss = criterion(outputs, labels)
        #calculate metrics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    #calculate epoch loss and accuracy
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.float() / len(test_loader.dataset)

    end_time = time.time() # end time of the epoch

    epoch_mins, epoch_secs, epoch_ms = epoch_step_time(start_time, end_time)

    #print metrics
    print(f"Test loss: {epoch_loss:.3f}.. ")
    print(f"Test accuracy: {epoch_acc:.3f}")
    print(f"Test Time: {epoch_mins}m {epoch_secs}s {epoch_ms}ms")

    #log metrics to comet_ml
    if experiment:
        inference_time = (end_time - start_time) / len(test_loader.dataset)
        experiment.log_metric("test_loss", epoch_loss)
        experiment.log_metric("test_accuracy", epoch_acc)
        experiment.log_metric("inference_time", inference_time)
        experiment.log_confusion_matrix(all_labels, all_preds)

        #close experiment
        experiment.end() if experiment else None

    #return metrics
    return epoch_loss, epoch_acc


#plot metrics
def plot_metrics(train_losses, train_acc, val_losses, val_acc):
    #detach tensors from gpu
    train_acc = [t.cpu().detach().numpy() for t in train_acc]
    val_acc = [t.cpu().detach().numpy() for t in val_acc]
    #plot training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    #plot training and validation accuracy
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.legend(frameon=False)
    plt.show()

# Delete .DS_Store files
def delete_ds_store(root_path):
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.DS_Store'):
                file_path = os.path.join(subdir, file)
                os.remove(file_path)

import os
import shutil
from glob import glob

def find_top_classes(input_folder, output_folder, n):
    class_counts = []
    
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if os.path.isdir(class_path):
            image_count = len(glob(os.path.join(class_path, '*.jpg'))) + len(glob(os.path.join(class_path, '*.png')))
            class_counts.append((class_name, image_count))

    class_counts.sort(key=lambda x: x[1], reverse=True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for class_name, _ in class_counts[:n]:
        src = os.path.join(input_folder, class_name)
        dst = os.path.join(output_folder, class_name)
        shutil.copytree(src, dst)

def create_class_folders(output_folder, class_name, train_folder, val_folder, test_folder):
    for folder in [train_folder, val_folder, test_folder]:
        class_path = os.path.join(folder, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

def split_images(output_folder, ratios):
    train_ratio, val_ratio, test_ratio = ratios
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    for class_name in os.listdir(output_folder):
        if class_name in ['train', 'val', 'test']:
            continue

        create_class_folders(output_folder, class_name, train_folder, val_folder, test_folder)

        class_folder = os.path.join(output_folder, class_name)
        images = glob(os.path.join(class_folder, '*.jpg')) + glob(os.path.join(class_folder, '*.png'))
        
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        for image_path in images[:train_end]:
            shutil.move(image_path, os.path.join(train_folder, class_name))
        
        for image_path in images[train_end:val_end]:
            shutil.move(image_path, os.path.join(val_folder, class_name))
            
        for image_path in images[val_end:]:
            shutil.move(image_path, os.path.join(test_folder, class_name))

        os.rmdir(class_folder)

if __name__ == "__main__":

    # Example usage:
    input_folder = 'data/VGG-Face2/data/train'
    output_folder = 'data/vgg_50'
    n = 50
    find_top_classes(input_folder, output_folder, n)

    ratios = [0.8, 0.1, 0.1]
    split_images(output_folder, ratios)
