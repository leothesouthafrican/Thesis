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
from image_attention_vis import VisualizeAttention
from PIL import Image
from torchsummary import summary
from thop import profile

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
    """_summary_: This function is used to load the model (state_dict) from the path if it exists/specified.
        Args:
            model: the model to be loaded
            path: the path where the model is saved
        Returns:
            model: the loaded model
    """
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
    """
    This function calculates the mean and standard deviation of images in the specified data path.
    
    Args:
        data_path (str): The path to the folder containing images.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the images.
    """
    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
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

def measure_flops_and_time(device, model):
    input_size = (3, 224, 224)
    model = model.to(device)
    device_str = str(device)  # Convert torch.device object to string

    input_data = torch.randn(1, *input_size).to(device)
    
    # Use thop to get FLOPs and number of parameters
    flops, params = profile(model, inputs=(input_data, ), verbose=False)

    return flops, params

#train the model
def train(model, train_loader, val_loader, criterion, optimizer,scheduler, hyper_params, verbose = 0, test_transform = None, experiment = False):

    #log hyperparameters
    experiment.log_parameters({key: val for key, val in hyper_params.items() if key != "model"}) if experiment else None

    #compute flops and params to compet
    flops, params = measure_flops_and_time(hyper_params["device"], model)
    
    experiment.log_metric("flops", flops) if experiment else None
    experiment.log_metric("params", params) if experiment else None

    #send model to device
    model.to(hyper_params["device"], dtype = torch.float32)

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
            step += 1

        #calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        #append metrics to lists
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        #validate model
        val_loss, vall_acc = validate(model, val_loader, criterion, hyper_params, verbose = verbose, test_transform = test_transform, experiment = experiment)

        #append metrics to lists
        val_losses.append(val_loss)
        val_acc.append(vall_acc)

        #change the learning rate
        scheduler.step(val_loss)

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
def validate(model, val_loader, criterion, hyper_params, verbose, test_transform = None, experiment = False):

    attention = VisualizeAttention(model, path = "/Users/leo/Programming/Thesis/data/att_viz_test/ImageNet_100_att/", hyper_params = hyper_params, transform = test_transform, experiment = experiment)

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

    #log images to comet_ml
    attention.log_images() if experiment else None

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


import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def process_images(input_path, output_path, n, split_ratio):
    # Step 1
    class_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    class_folders = sorted(class_folders, key=lambda x: len(os.listdir(os.path.join(input_path, x))), reverse=True)[:n]

    # Step 2
    avg_dimensions = {}
    for folder in class_folders:
        images = [f for f in os.listdir(os.path.join(input_path, folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        dimensions = []
        for img in images:
            im = Image.open(os.path.join(input_path, folder, img))
            dimensions.append(im.size)
        avg_dimensions[folder] = np.mean(dimensions, axis=0)
        print(f'{folder}: {avg_dimensions[folder]}')

    # Step 3
    moved_counts = {}
    for folder in class_folders:
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)
        images = [f for f in os.listdir(os.path.join(input_path, folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        dimensions = [Image.open(os.path.join(input_path, folder, img)).size for img in images]
        std_dev = np.std(dimensions, axis=0)
        mean_dim = avg_dimensions[folder]
        moved, left = 0, 0
        for img, dim in zip(images, dimensions):
            if (dim[0] > mean_dim[0] - std_dev[0]) and (dim[1] > mean_dim[1] - std_dev[1]):
                shutil.copy(os.path.join(input_path, folder, img), os.path.join(output_path, folder, img))
                moved += 1
            else:
                left += 1
        moved_counts[folder] = (moved, left)
        print(f'{folder}: {moved} moved, {left} left behind')

    # Step 4
    for folder in class_folders:
        images = [f for f in os.listdir(os.path.join(output_path, folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_images, test_images = train_test_split(images, test_size=split_ratio[2], random_state=42)
        train_images, val_images = train_test_split(train_images, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]), random_state=42)
        
        for img_type, img_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            os.makedirs(os.path.join(output_path, img_type, folder), exist_ok=True)
            for img in img_list:
                shutil.move(os.path.join(output_path, folder, img), os.path.join(output_path, img_type, folder, img))

    # Step 5
    for folder in class_folders:
        shutil.rmtree(os.path.join(output_path, folder))

# Example usage:
if __name__ == '__main__':
    input_path = '/Users/leo/Desktop/TinyNet/ILSVRC/Data/CLS-LOC/train/'
    output_path = '/Users/leo/Desktop/Thesis/data/ImageNet_100/'
    n = 100
    split_ratio = [0.8, 0.1, 0.1]
    process_images(input_path, output_path, n, split_ratio)
