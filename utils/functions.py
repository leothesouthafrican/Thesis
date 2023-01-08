import time
import torch
import torch.nn as nn
from tqdm import tqdm
from comet_ml import Experiment
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from PIL import Image
import concurrent.futures

def display_filters(state_dict, experiment, channels=1):
    # Extract the weights of the convolutional layers from the state dictionary
    filters1 = None
    for key, value in state_dict.items():
        if 'conv1' in key and 'weight' in key:
            filters1 = value
    
    # Copy the filters to the host memory
    filters1 = filters1.cpu().numpy()

    # Calculate the number of rows and columns based on the number of filters
    num_filters = filters1.shape[0]
    num_cols = 5
    num_rows = (num_filters // num_cols) + (num_filters % num_cols != 0)

    # Display the filters using Matplotlib
    plt.figure(figsize=(20, 10))
    for i in range(num_filters):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(filters1[i][0], cmap='gray')
        plt.axis('off')

    # Save the plot to the experiment using log_figure
    experiment.log_figure(figure=plt.gcf(), figure_name='filters_1.png')

    # Display the plot (optional)
    plt.show()

    return filters1


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def _weights_init(m):
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
    return acc

# Training Function 
def train(num_epochs, model, loss_fn, optimizer, train_loader, val_loader, best_model_path, device, experiment): 

    with experiment.train():
        best_accuracy = 0.0 

        #setting the model to train mode
        model.train()
        print("Begin training...") 
        for epoch in range(1, num_epochs+1): 
            running_train_loss = 0.0 # training loss
            running_accuracy = 0.0  # validation accuracy
            running_vall_loss = 0.0  # validation loss
            total = 0 # total number of samples
            steps = 0 # number of batches
            start_time = time.time() # start time of the epoch

            #avoiding unbound variable error
            train_loss = 0
            train_acc = 0

            # Training Loop 
            for x, y in tqdm(train_loader):
                step_start_time = time.time() # start time of the batch
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()   # zero the parameter gradients          
                y_pred = model(x)   # predict output from the model 
                train_loss = loss_fn(y_pred, y)   # calculate loss for the predicted output

                # Calculate the training accuracy
                train_acc = calculate_accuracy(y_pred, y)

                train_loss.backward()   # backpropagate the loss 
                optimizer.step()        # adjust parameters based on the calculated gradients 
                running_train_loss +=train_loss.item()  # track the loss value

                step_end_time = time.time() # end time of the batch

                # Log the metrics to Comet.ml
                experiment.log_metrics({
                    "loss": train_loss.item(),
                    "acc": train_acc.item(),
                    'step_time': epoch_step_time(step_start_time, step_end_time)[1]
                    }
                    ,step=steps, epoch=epoch)
                steps += 1 

            # Validation Loop 
            with torch.no_grad(): 
                model.eval() 
                for x, y in tqdm(val_loader): 
                    
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                
                    val_loss = loss_fn(y_pred, y) 
                
                    # The label with the highest value will be our prediction 
                    _, predicted = torch.max(y_pred, 1) 
                    running_vall_loss += val_loss.item() # track the loss value
                    total += y.size(0) # track the total number of samples
                    running_accuracy += (predicted == y).sum().item() 

            # Calculate validation loss value 
            val_loss_value = running_vall_loss/len(val_loader) 

            # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
            accuracy = running_accuracy / total    

            # Save the model if the accuracy is the best 
            if accuracy > best_accuracy: 
                torch.save(model.state_dict(), best_model_path)
                best_accuracy = accuracy
                display_filters(model.state_dict(), experiment) #logs the filters to Comet.ml
                print("Filters saved")

            # Log the metrics to Comet.ml
            experiment.log_metrics({
                "val_loss": val_loss_value,
                "val_acc": accuracy
                }
                ,step=epoch)

            end_time = time.time() # end time of the epoch

            # Calculate the time taken for the epoch
            epoch_mins, epoch_secs = epoch_step_time(start_time, end_time)

            # Print the statistics of the epoch 
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc *100:.2f}%')
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Val. Loss: {val_loss_value:.3f} |  Val. Acc: {accuracy*100:.2f}%')

#Evaluation Function
def evaluate(model, iterator, criterion, device, experiment):
    
    epoch_loss = 0
    epoch_acc = 0

    model.eval() #Setting the model to evaluation mode

    with torch.no_grad(): #Turning off gradient calculation

        for (x, y) in tqdm(iterator):

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()

            epoch_acc += acc.item()

            total_loss = epoch_loss / len(iterator)
            total_acc = epoch_acc / len(iterator)

            # Log the metrics to Comet.ml
            experiment.log_metrics({
                "loss": total_loss,
                "acc": total_acc*100
                })

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def confusion(model,test_loader, experiment, device):
    
    #Get the predictions
    y_pred = []
    y_true = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred.append(model(x).argmax(1).cpu())
            y_true.append(y.cpu())

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    #Get the confusion matrix
    experiment.log_confusion_matrix(y_pred, y_true)



class ImageDataset_CSV:
    """
    A class that takes a folder of images used in a classification task and returns a csv file
    with data pertaining to the image as well as the image itself in an array format.
    """
    def __init__(self, root_folder_path, output_file_path, img_size=224):
        self.root_folder_path = root_folder_path
        self.output_file_path = output_file_path
        self.img_size = img_size

    def get_image_data(self):
        """Returns a list of dictionaries with the data of each image in the folder.
        """
        image_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Create a list of tasks to be executed concurrently
            tasks = [executor.submit(self.get_image_data_for_folder, folder) for folder in os.listdir(self.root_folder_path)]

            # Iterate over the tasks and collect the results
            for task in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks), desc='Processing images in {}'.format(self.root_folder_path)):
                image_data.extend(task.result())

        return image_data

    def get_image_data_for_folder(self, folder):
        """Returns a list of dictionaries with the data of each image in the given folder.
        """
        data = []
        for file in os.listdir(os.path.join(self.root_folder_path, folder)):
            data.append({
                'image_path': os.path.join(self.root_folder_path, folder, file),
                'image_label': folder,
                'image_array': self.get_image_array(os.path.join(self.root_folder_path, folder, file))
            })
        return data

    def get_image_array(self, image_path):
        """Returns an array of the image.
        """
        img = Image.open(image_path)
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)
        return img



    def create_csv(self):
        """Creates a csv file with the data of each image in the folder without pandas
        """

        # Get the image data
        image_data = self.get_image_data()

        # Create the csv file
        with open(self.output_file_path, 'w') as csv_file:
            # Create the header
            csv_file.write('image_path,image_label,')
            for i in range(self.img_size * self.img_size * 3):
                csv_file.write('pixel_{},'.format(i))
            csv_file.write(' ')

            # Write the data in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Create a list of tasks to execute in parallel
                tasks = [executor.submit(self._write_image_data, csv_file, image) for image in image_data]
                # Iterate over the tasks and display the progress using tqdm
                for f in tqdm(concurrent.futures.as_completed(tasks),total = len(tasks), desc='Writing to {}'.format(self.output_file_path)):
                    pass

    def _write_image_data(self, csv_file, image):
        """Write the data for a single image to the csv file"""
        csv_file.write('{},{},'.format(image['image_path'], image['image_label']))
        for pixel in image['image_array'].flatten():
            csv_file.write('{},'.format(pixel))
        csv_file.write(' ')



if __name__ == '__main__':
    # Create a Comet.ml experiment
    
    create_dataset = ImageDataset_CSV('data/train', 'data/train.csv')
    #create_dataset.get_image_data()
    create_dataset.create_csv()