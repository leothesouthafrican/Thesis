import time
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor

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
def train(model,loss_fn, optimizer, hyper_params, train_loader, val_loader): 

    with hyper_params['comet_exp'].train():
        best_accuracy = 0.0 

        #setting the model to train mode
        model.train()
        print("Begin training...") 
        for epoch in range(1, hyper_params['epochs']+1):
            running_train_loss = 0.0 # training loss
            running_accuracy = 0.0  # validation accuracy
            running_val_loss = 0.0  # validation loss
            total = 0 # total number of samples
            steps = 0 # number of batches
            start_time = time.time() # start time of the epoch

            #avoiding unbound variable error
            train_loss = 0
            train_acc = 0

            # Training Loop 
            for x, y in tqdm(train_loader):
                step_start_time = time.time() # start time of the batch
                x = x.to(hyper_params['device'])
                y = y.to(hyper_params['device'])
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
                hyper_params['comet_exp'].log_metrics({
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

                    x = x.to(hyper_params['device'])
                    y = y.to(hyper_params['device'])
                    y_pred = model(x)

                    val_loss = loss_fn(y_pred, y) 

                    # The label with the highest value will be our prediction 
                    _, predicted = torch.max(y_pred, 1) 
                    running_val_loss += val_loss.item() # track the loss value
                    total += y.size(0) # track the total number of samples
                    running_accuracy += (predicted == y).sum().item() 

            # Calculate validation loss value 
            val_loss_value = running_val_loss/len(val_loader) 

            # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
            accuracy = running_accuracy / total    

            # Save the model if the accuracy is the best 
            if accuracy > best_accuracy: 
                torch.save(model.state_dict(), hyper_params['best_model_path'])
                best_accuracy = accuracy
                print(f'Best model saved at epoch {epoch + 1} with accuracy {round(accuracy,4)} and loss {round(val_loss_value,4)}.')

            elif hyper_params['save_at_end'] and epoch == hyper_params['epochs']:
                file_name = hyper_params['best_model_path'].split('.')[0]
                suffix = hyper_params['best_model_path'].split('.')[1]
                torch.save(model.state_dict(), f'{file_name}_end.{suffix}')

            # Log the metrics to Comet.ml
            hyper_params['comet_exp'].log_metrics({
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
def evaluate(model, test_loader, criterion, device, experiment):
    
    with experiment.test():

        epoch_loss = 0
        epoch_acc = 0
        model.eval() #Setting the model to evaluation mode
        with torch.no_grad(): #Turning off gradient calculation
            for (x, y) in tqdm(test_loader):

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                loss = criterion(y_pred, y)
                acc = calculate_accuracy(y_pred, y)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                total_loss = epoch_loss / len(test_loader)
                total_acc = epoch_acc / len(test_loader)

                # Log the metrics to Comet.ml
                experiment.log_metrics({
                    "loss": total_loss,
                    "acc": total_acc*100
                    })

    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)

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

def get_kaggle_data(api_token, dataset_name, data_destination):
    """Download a dataset from Kaggle using the API token"""
    # Create a kaggle directory in the home directory
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
    
    # Create a kaggle.json file with the API token
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json, "w") as f:
        json.dump(api_token, f)

    # Download the dataset
    command = f"kaggle datasets download -d {dataset_name} -p {data_destination}"

    # Run the command
    os.system(command)

    # Unzip the files
    unzip_files(data_destination)
    
def unzip_files(directory):
    """Takes a directory path and unzips all the zip files in the directory using parallel processing"""
    # Get a list of all the zip files in the directory
    zip_files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    
    if not zip_files:
        print("No zip files found in the directory.")
        return
    
    # Unzip the files using a process pool
    with ProcessPoolExecutor() as executor:
        for zip_file in zip_files:
            path = os.path.join(directory, zip_file)
            executor.submit(unzip, path)
            
def unzip(path):
    """Unzip a single zip file"""
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(path))