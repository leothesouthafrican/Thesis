import os

def image_splitter(data_path, train_val_test_split):

    train_val_test = train_val_test_split

    # create train, val and test folders
    os.mkdir(os.path.join(data_path, "train"))
    os.mkdir(os.path.join(data_path, "val"))
    os.mkdir(os.path.join(data_path, "test"))

    # create folders for each class inside train, val and test folders
    for folder in os.listdir(data_path):

        #ignore .DS_Store
        if folder == ".DS_Store":
            continue

        if folder == "train" or folder == "val" or folder == "test":
            continue

        os.mkdir(os.path.join(data_path, "train", folder))
        os.mkdir(os.path.join(data_path, "val", folder))
        os.mkdir(os.path.join(data_path, "test", folder))

        # get all images in the folder
        images = os.listdir(os.path.join(data_path, folder))

        # split images into train, val and test
        train_images = images[:int(len(images) * train_val_test[0])]
        val_images = images[int(len(images) * train_val_test[0]):int(len(images) * (train_val_test[0] + train_val_test[1]))]
        test_images = images[int(len(images) * (train_val_test[0] + train_val_test[1])):]

        # move images to train, val and test folders
        for image in train_images:
            os.rename(os.path.join(data_path, folder, image), os.path.join(data_path, "train", folder, image))

        for image in val_images:
            os.rename(os.path.join(data_path, folder, image), os.path.join(data_path, "val", folder, image))

        for image in test_images:
            os.rename(os.path.join(data_path, folder, image), os.path.join(data_path, "test", folder, image))

        # remove original folder
        os.rmdir(os.path.join(data_path, folder))

if __name__ == "__main__":
    data_path = "/Users/leo/Desktop/new_thesis/data/faces_50/"
    train_val_test_split = [0.8, 0.15, 0.05]
    image_splitter(data_path, train_val_test_split)

