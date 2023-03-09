import time
import shutil
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

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
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Function to calculate the accuracy of the model
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True) #get the index of the max log-probability
    correct = top_pred.eq(y.view_as(top_pred)).sum() #get the number of correct predictions
    acc = correct.float() / y.shape[0] #calculate the accuracy

    #save to txt file
    np.savetxt('y_pred_training.txt', y_pred.detach().cpu().numpy(), delimiter=',')
    np.savetxt('y_pred_top.txt', top_pred.detach().cpu().numpy(), delimiter=',')
    return acc

def MBNV3_build(num_classes, model, weights, module, device):
    
    #load the model
    model = model(weights = weights)
    print(f"Model loaded with {weights} weights")

    #change the last layer to output specified number of classes
    model.classifier[3] = nn.Linear(model.classifier[-1].in_features, num_classes, bias=True)
    print(f"Output layer modified to output {num_classes} classes")

    if module != None:
        #replace the SE block with CBAM
        counter = 0
        for i in range(len(model.features)):
            try:
                if type(model.features[i].block[2]) == torchvision.ops.misc.SqueezeExcitation:
                    #get the output shape of the layer before the SE block
                    prev_out_channels = model.features[i].block[0].out_channels
                    #replace the SE block with CBAM
                    model.features[i].block[2] = module(prev_out_channels)
                    counter += 1
            except:
                pass
        print(f"{counter} SE blocks replaced with {module}")
    
    #freeze the weights of the model except the last layer and the inserted module layers
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier[-1].parameters():
        param.requires_grad = True
    print("Weights of the model frozen except the last layer and the inserted module layers")

    #init the weights of the last layer
    model.classifier[3].apply(_weights_init)
    print("Weights of the last layer initialized")

    #init the weights of the inserted module layers or the SE block
    if module != None:
        for i in range(len(model.features)):
            try:
                if type(model.features[i].block[2]) == module:
                    #init the weights of the inserted module layers
                    model.features[i].block[2].apply(_weights_init)
                    for param in model.features[i].block[2].parameters():
                        param.requires_grad = True
            except:
                pass
        print("Weights of the inserted module layers initialized and weights trainable")
    else:
        for i in range(len(model.features)):
            try:
                if type(model.features[i].block[2]) == torchvision.ops.misc.SqueezeExcitation:
                    #init the weights of the SE block
                    model.features[i].block[2].apply(_weights_init)
                    for param in model.features[i].block[2].parameters():
                        param.requires_grad = True
            except:
                pass
        print("Weights of the SE block initialized and weights trainable")

    return model.to(device)

#train the model
def train(model, train_loader, val_loader, criterion, optimizer, hyper_params, verbose = 0):

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
        for inputs, labels in tqdm(train_loader, disable= True if verbose == 0 else False):
            #send inputs and labels to device
            inputs = inputs.to(hyper_params["device"])
            labels = labels.to(hyper_params["device"])
            #set gradients to zero
            optimizer.zero_grad()
            #forward pass
            outputs = model(inputs)
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
        epoch_mins, epoch_secs = epoch_step_time(start_time, end_time)

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
def test(model, test_loader, criterion, hyper_params):

    start_time = time.time() # start time of the epoch

    #load the best model
    model.load_state_dict(torch.load(hyper_params['model_save_path']))
    #set model to evaluation mode
    model.eval()
    #initialise variables to store metrics
    running_loss = 0.0
    running_corrects = 0
    #loop through batches
    for inputs, labels in test_loader:
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
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.float() / len(test_loader.dataset)

    end_time = time.time() # end time of the epoch

    epoch_mins, epoch_secs = epoch_step_time(start_time, end_time)

    #print metrics
    print(f"Test loss: {epoch_loss:.3f}.. ")
    print(f"Test accuracy: {epoch_acc:.3f}")
    print(f"Test Time: {epoch_mins}m {epoch_secs}s")
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