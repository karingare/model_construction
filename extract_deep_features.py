
import torch
from torchvision import transforms, models
from torch import nn
import csv
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
import argparse
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd
import ast
import numpy as np
from skbio.stats.ordination import pcoa
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from supportive_code.data_setup import create_dataloaders
from torchvision.models.feature_extraction import get_graph_node_names
from supportive_code.padding import NewPad

if __name__ == "__main__":  
    ###################### Model setup ##########################
    # This section imports a previously trained neural network, along with information about the classes 

    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")


    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (test )', default='syke2022')
    parser.add_argument('--num_epochs', type=int, help='Specify data selection (test or all)', default=20)
    parser.add_argument('--model', type=str, help='Specify model (name of model of main)', default='syke2022')
    parser.add_argument('--renew', type=str, help='Specify if features should be extracted from start (yes or no)', default='no')
    parser.add_argument('--samplingtype', type=str, help='Specify how subsampling should be done (random10, bytaxa, none)', default='bytaxa')
    parser.add_argument('--taxa', type=str, help='Specify taxa to look at (diatoms or cyanobacteria))', default='diatoms')
    parser.add_argument('--pcoaandtsne', type=str, help='Specify taxa to look at (diatoms or cyanobacteria))', default='yes')

    args = parser.parse_args()

    if parser.parse_args().data == "test":
        data_path = base_dir / 'data' / 'development'
        unclassifiable_path = base_dir / 'data' / 'development_unclassifiable'
    elif parser.parse_args().data == "smhibaltic2023":
        data_path = base_dir / 'data' / 'smhi_training_data_oct_2023' / 'Baltic'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "syke2022":
        data_path = base_dir / 'data' / 'SYKE_2022' / 'labeled_20201020'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # import the model
    if parser.parse_args().model == "main":
        model_path = base_dir / 'data' / 'models' /'model_main_240116' 
    elif parser.parse_args().model == "development":
        model_path = base_dir / 'data' / 'models' / 'development_20240209' 
    elif parser.parse_args().model == "syke2022":
        model_path = base_dir / 'data' / 'models' / 'syke2022_20240227'
    else:
        model_path = base_dir / 'data' / 'models' / parser.parse_args().model 

    figures_path =  model_path / 'figures'
    path_to_model = model_path / 'model.pth'

    with open(model_path / 'class_to_idx.txt') as f:
            data = f.read()
    

    training_info_path = model_path / 'training_info.txt'

    # Read the file contents
    training_info = {}
    with open(training_info_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)
            # Try to evaluate value if it's a list or int/float, otherwise keep it as a string
            try:
                value = eval(value)
            except (SyntaxError, NameError):
                pass
            training_info[key] = value

    # Example: Access the padding_mode
    padding_mode = training_info.get('padding_mode')



    # Infromation about the classes and dicts to translate between the class index (a number) and the class name (ex: Thalassiosira_levanderi)
    class_to_idx = ast.literal_eval(data)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes_in_model_total = len(class_to_idx)
    class_names = list(class_to_idx.keys())


    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes_in_model_total)
    ) 
    model.load_state_dict(torch.load(path_to_model))


    ########################## Creation of dataloader ##########################

    padding = True # controls padding of images

    # selection of data to load
    BATCH_SIZE = 32

    # image transforms used
    simple_transform = transforms.Compose([
            NewPad(padding_mode=padding_mode),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    _, dataloader1, _, dataloader2, _, class_names, class_to_idx = create_dataloaders( 
        data_path = data_path,
        transform = simple_transform,
        unclassifiable_path=unclassifiable_path,
        simple_transform = simple_transform,
        batch_size = BATCH_SIZE
    )








    ############################ Setting up feature extration ##########################

    model = nn.Sequential(*list(model.children())[:-1])

    model.to(device)

    # nodes, _ = get_graph_node_names(model)

    def extract_features(model, dataloader):
        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, class_labels in dataloader:
                # Forward pass up to AdaptiveAvgPool2d
                outputs = model(inputs.to(device))
                features.append(outputs.squeeze().cpu().numpy())
                labels.extend(class_labels)

        # Convert the array to a DataFrame
        feature_columns = [f'feature_{i}' for i in range(features[0].shape[1])]
        features_df = pd.DataFrame(data=np.vstack(features), columns=feature_columns)
        labels_df = pd.DataFrame(data={'class': labels})
        labels_df['class'] = labels_df['class'].astype(int)

        return features_df, labels_df

    ########################## Extract features from ALL data in the datasets ##########################
    # File paths
    features_file_path = base_dir / 'data' / 'deep_feature_extraction' / 'deep_features.csv'
    labels_file_path = base_dir / 'data' / 'deep_feature_extraction' / 'labels.csv'
    test_features_file_path = base_dir / 'data' / 'deep_feature_extraction' / 'test_deep_features.csv'
    test_labels_file_path = base_dir / 'data' / 'deep_feature_extraction' / 'test_labels.csv'

    all_paths_exist = all(os.path.exists(path) for path in [features_file_path, labels_file_path, test_features_file_path, test_labels_file_path])


    if args.renew == 'yes' or not all_paths_exist:
        print('Extracting fresh features... ')
        # Files don't exist, call the function to extract deep features
        deep_features, labels = extract_features(model, dataloader1)
        
        # Save the extracted features and labels
        deep_features.to_csv(features_file_path, index=False)
        labels.to_csv(labels_file_path, index=False)
        labels = labels['class']

        # do the same thing for a test set
        test_deep_features, test_labels = extract_features(model, dataloader2)

        # save the extracted features and labels
        test_deep_features.to_csv(test_features_file_path, index=False)
        test_labels.to_csv(test_labels_file_path, index=False)
        test_labels = test_labels['class']
    else: 
        print('Reading features...')
        # Files exist, read them
        deep_features = pd.read_csv(features_file_path)
        labels = pd.read_csv(labels_file_path)
        labels = labels['class']

        test_deep_features = pd.read_csv(test_features_file_path)
        test_labels = pd.read_csv(test_labels_file_path)
        test_labels = test_labels['class']

    labels = labels.map(idx_to_class)
    test_labels = test_labels.map(idx_to_class)

    data = pd.concat([deep_features, labels], axis=1)
    test_data  = pd.concat([test_deep_features, test_labels], axis=1)








    ########################## CRETAING SAMPLERS FOR SUBSETS ##########################

    # creating functions for sampling

    def custom_sampler(data, n_samples_per_class, n_classes):
        # Randomly choose n_classes from the classes in data
        selected_classes = np.random.choice(data['class'].unique(), n_classes, replace=False)
        
        # A function to get a number of samples from each class
        def get_samples(group):
            return group.sample(n_samples_per_class, replace=True)
        
        # Filter the data to include only the selected classes
        subset_data = data[data['class'].isin(selected_classes)]

        # Apply the function to each group within the selected classes
        sampled_data = subset_data.groupby('class', group_keys=False).apply(get_samples)
        return sampled_data

    def targetted_sampler(data, class_names, even, n_samples_per_class=0):
        # Filter the data to include only the specified class names
        subset_data = data[data['class'].isin(class_names)]
        
        sampled_data = subset_data
        if even == True: 
            # Use the same function as in the pther sampler
            def get_samples(group):
                return group.sample(n_samples_per_class, replace=True)
            
            # Apply the function to each group within the selected classes
            sampled_data = subset_data.groupby('class', group_keys=False).apply(get_samples)
        return sampled_data





    ########################## CHOOSING THE SUBSETS ##########################

    if args.samplingtype != 'none':
        print('Selecting a subset...')

        if args.samplingtype == 'random10':
            ########################## SUBSETTING - OPTION 1 ##########################
            # selects 10 random taxa and 100 instances of each 
            n_samples_per_class = 100
            n_classes = 10
            # Use the function to create the subset
            subset_data = custom_sampler(data, n_samples_per_class, n_classes)
            test_subset_data = custom_sampler(test_data, n_samples_per_class, n_classes) # gets n_samples_per_class samples from another dataset

        elif args.samplingtype == 'bytaxa' and args.taxa == 'none':
            print('Ypu need to specify a --taxa for this to work :) ')

        elif args.samplingtype == 'bytaxa':
            ########################## SUBSETTING - OPTION 2 ##########################

            # OPTION 2 is working on a subset defined by taxonomy. So far, I've worked with diatoms and cyanobacteria.

            # Read some taxonomic info
            tax_info = pd.read_excel(base_dir /'model_construction' / 'supportive_files' / 'dictionary.xlsx')

            if args.taxa == 'cyanobacteria':
                selected_classes = tax_info.loc[tax_info['taxonomic_class'] == 'Cyanophyceae', 'assigned_class']

            elif args.taxa == 'diatoms':
                selected_classes = tax_info.loc[tax_info['taxonomic_class'].isin(['Bacillariophyceae', 'Mediophyceae']), 'assigned_class']

            n_classes = len(selected_classes)
            n_samples_per_class_test = 20

            # Call the targetted_sampler functions
            subset_data = targetted_sampler(data=data, class_names=selected_classes, even=False) # gets all data of the selected classes
            test_subset_data = targetted_sampler(data=test_data, class_names=selected_classes, even=False) # gets n_samples_per_class samples from another dataset
            
        else: 
            print('Invalid subsetting method. Choose bytaxa or random10')

        ########################## SUBSETTING - LAST STEP ##########################
        # Split the subset data into features and labels
        deep_features_subset = subset_data.drop('class', axis=1)
        labels_subset = subset_data['class']

        test_deep_features_subset = test_subset_data.drop('class', axis=1)
        test_labels_subset = test_subset_data['class']


    else:
        # note: if there is no subsetting, these variables do not carry subset but they carry the entire dataset. 
        # its not pretty but it keeps the code running :)
        deep_features_subset = data.drop('class', axis=1)
        labels_subset = data['class']

        test_deep_features_subset = test_data.drop('class', axis=1)
        test_labels_subset = test_data['class']
        n_classes = len(set(labels_subset))


    print('\n These are the taxa we are looking at:\n')
    print(labels_subset.unique())

    print(f'The number of classes is {n_classes}')



    ######## remove !!!!!!
    deep_features_subset = test_deep_features_subset
    labels_subset = test_labels_subset


    ########################## SETTING UP t-SNE AND PCoA ##########################

    if args.pcoaandtsne == 'yes': 
        print('Setting up PCoA...')
        # Min-Max Scaling
        scaler = MinMaxScaler()
        deep_features_scaled = scaler.fit_transform(deep_features_subset)

        # Compute the dissimilarity matrix using Euclidean distance
        dissimilarity_matrix = pairwise_distances(deep_features_scaled, metric='euclidean')

        # Symmetrize the dissimilarity matrix
        dissimilarity_matrix = 0.5 * (dissimilarity_matrix + dissimilarity_matrix.T)

        # Set diagonal elements to 0
        np.fill_diagonal(dissimilarity_matrix, 0)

        # Save the dissimilarity matrix
        np.savetxt(base_dir / 'data' / 'deep_feature_extraction' / 'dissimilarity_matrix_scaled.csv', dissimilarity_matrix, delimiter=',')

        ## to-do: are all these steps really needed?

        # Perform PCoA
        ordination_results = pcoa(dissimilarity_matrix)

        print('Setting up t-SNE...')
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(dissimilarity_matrix)

        ############################ SAVE FILES for visualization on my laptop  ##########################

        path_for_laptop = base_dir / 'data' / 'deep_feature_extraction' / 'files_to_laptop'

        # Save ordination_results
        joblib.dump(ordination_results, path_for_laptop / 'ordination_results.joblib')

        # Save labels_subset
        np.save(path_for_laptop / 'labels_subset.npy', labels_subset)

        # Save n_classes
        np.save(path_for_laptop / 'n_classes.npy', np.array([n_classes]))

        # Save t-SNE results
        joblib.dump(tsne_results, path_for_laptop / 'tsne_results.joblib')








    ############################ SETTING UP GAUSSIAN MIXTURE MODEL  ##########################

    print('Setting up GMM... ')

    # Standardize the features  - important for GMMs
    # this is a different type of scaling compared to the first scaling (that was min max)
    scaler = StandardScaler()
    scaler = scaler.fit(deep_features_subset)
    deep_features_subset_scaled = scaler.transform(deep_features_subset)


    # Initialize Gaussian Mixture Model with the same number of clusters as we had number of selected classes
    n_components = n_classes*100
    gmm = GaussianMixture(n_components=n_components, init_params='kmeans')

    # Fit the GMM
    gmm.fit(deep_features_subset_scaled)

    if gmm.converged_:
        print("The GMM converged :)")
    else:
        print("The GMM did not converge :(")


    ############################ PREDICT USING THE GMM AND TRUE LABELS FOR TRAINING SET ##########################
    # The prupose of this is to connect each cluster in the gmm to a class of plankton
    # it uses majority vote, so the most common label in each cluster will be the label predicted to future instances in that cluster

    # Get the predicted labels for each sample (these are the cluster numbers)
    predicted_labels = gmm.predict(deep_features_subset_scaled)

    labels_subset_numeric = [class_to_idx[value] for value in labels_subset] # the true labels in numeric format

    cluster_to_label = {} # a dictionary to store mapping

    for pred_label in range(0, n_components):
        # 1: find indices of instances with the current predicted label
        indices_with_pred_label = [i for i, label in enumerate(predicted_labels) if label == pred_label]

        # 2: Extract the true labels for instances with the current predicted label
        true_labels_with_pred_label = [labels_subset_numeric[i] for i in indices_with_pred_label]

        # 3: find the most common of the true labels
        most_common_true_label = max(set(true_labels_with_pred_label), key=true_labels_with_pred_label.count)

        # 4: update the dictionary with the mapping
        cluster_to_label[pred_label] = most_common_true_label

    temp = predicted_labels

    # Map predicted labels to numeric classes
    predicted_labels = [cluster_to_label[value] for value in temp] # updates so these numbers mean the same as the numbers in the other scripts
    pred_classes = [idx_to_class[value] for value in predicted_labels] # the class names, ex Skeletonema marinoi

    ############################ MAKE CONFUSION MATRIX FOR TRAIN SET ##########################

    # Get unique class labels
    unique_labels_set = set(labels_subset)
    unique_labels = sorted(list(unique_labels_set))

    # Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Initialize confusion matrix with empty columns
    test_conf_matrix = [[0 for _ in unique_labels] for _ in unique_labels]

    # Fill in the confusion matrix
    for true_label, pred_class in zip(test_labels_subset, pred_classes):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_class]
        test_conf_matrix[true_index][pred_index] += 1

    # Save testing confusion matrix to CSV
    test_confusion_file_path = path_for_laptop / "train_confusion_matrix.csv"

    with open(test_confusion_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["True Label"] + unique_labels)
        # Write data
        for i, row in enumerate(test_conf_matrix):
            writer.writerow([unique_labels[i]] + row)



    # the data in this matrix is commonly imbalanced as the data is very imbalanced. ex: there is a lot of chaetoceros



    ############################ TESTING THE PERFORMANCE WITH NEW DATA = TEST SET ##########################

    print('Predicting on test set with GMM... ')

    # scale with same scaler as before
    test_deep_features_subset_scaled = scaler.transform(test_deep_features_subset)

    # Get the predicted labels for each sample. first use the model
    test_predicted_labels = gmm.predict(test_deep_features_subset_scaled)

    # and then the dictionaries
    test_predicted_labels = [cluster_to_label[value] for value in test_predicted_labels]
    pred_classes = [idx_to_class[value] for value in test_predicted_labels] # with names

    # get the true classes of the test set in numeric format
    test_true_labels = [class_to_idx[value] for value in test_labels_subset]

    ############################ MAKE CONFUSION MATRIX FOR TEST SET ##########################

    # Get unique class labels
    unique_labels_set = set(test_labels_subset)
    unique_labels = sorted(list(unique_labels_set))

    print(unique_labels)
    # Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Initialize confusion matrix with empty columns
    test_conf_matrix = [[0 for _ in unique_labels] for _ in unique_labels]

    # Fill in the confusion matrix
    for true_label, pred_class in zip(test_labels_subset, pred_classes):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_class]
        test_conf_matrix[true_index][pred_index] += 1

    # Save testing confusion matrix to CSV
    test_confusion_file_path = path_for_laptop / "test_confusion_matrix.csv"

    with open(test_confusion_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["True Label"] + unique_labels)
        # Write data
        for i, row in enumerate(test_conf_matrix):
            writer.writerow([unique_labels[i]] + row)

    ############################ CALCULATE SOME METRICS ##########################

    ari_score = adjusted_rand_score(test_labels_subset, pred_classes)
    print(f"\nAdjusted Rand Index: {ari_score:.4f}")

    ########################### SAVE FILES ############################################

    # for the training set
    np.save(path_for_laptop / 'predicted_labels.npy', pred_classes)
    np.save(path_for_laptop / 'true_labels.npy', labels_subset)

    # for the testing set
    np.save(path_for_laptop / 'test_predicted_labels.npy', test_predicted_labels)
    np.save(path_for_laptop / 'test_true_labels.npy', test_labels_subset)


    ############################ SETTING UP K-MEANS ##########################

    print('Setting up KMeans... ')

    # Initialize KMeans with the same number of clusters as we had in the gmm
    kmeans = KMeans(n_clusters=n_components, init='k-means++', random_state=42)

    # Fit the KMeans model
    kmeans.fit(deep_features_subset_scaled)

    ############################ PREDICT USING KMEANS AND TRUE LABELS FOR TRAINING SET ##########################

    predicted_labels = kmeans.predict(deep_features_subset_scaled)

    labels_subset_numeric = [class_to_idx[value] for value in labels_subset] # the true labels in numeric format

    cluster_to_label = {} # a dictionary to store mapping

    for pred_label in range(0, n_components):
        # 1: find indices of instances with the current predicted label
        indices_with_pred_label = [i for i, label in enumerate(predicted_labels) if label == pred_label]

        # 2: Extract the true labels for instances with the current predicted label
        true_labels_with_pred_label = [labels_subset_numeric[i] for i in indices_with_pred_label]

        # 3: find the most common of the true labels
        most_common_true_label = max(set(true_labels_with_pred_label), key=true_labels_with_pred_label.count)

        # 4: update the dictionary with the mapping
        cluster_to_label[pred_label] = most_common_true_label

    temp = predicted_labels

    # Map predicted labels to numeric classes
    predicted_labels = [cluster_to_label[value] for value in temp] # updates so these numbers mean the same as the numbers in the other scripts
    pred_classes = [idx_to_class[value] for value in predicted_labels] # the class names, ex Skeletonema marinoi

    ############################ MAKE CONFUSION MATRIX FOR TRAIN SET (KMEANS) ##########################

    # Get unique class labels
    unique_labels_set = set(labels_subset)
    unique_labels = sorted(list(unique_labels_set))

    # Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Initialize confusion matrix with empty columns
    test_conf_matrix = [[0 for _ in unique_labels] for _ in unique_labels]

    # Fill in the confusion matrix
    for true_label, pred_class in zip(test_labels_subset, pred_classes):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_class]
        test_conf_matrix[true_index][pred_index] += 1

    # Save testing confusion matrix to CSV
    test_confusion_file_path = path_for_laptop / "train_confusion_matrix_kmeans.csv"

    with open(test_confusion_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["True Label"] + unique_labels)
        # Write data
        for i, row in enumerate(test_conf_matrix):
            writer.writerow([unique_labels[i]] + row)





    ############################ TESTING THE PERFORMANCE WITH NEW DATA = TEST SET KMEANS ##########################

    print('Predicting on test set with  Kmeans... ')

    # scale with same scaler as before
    test_deep_features_subset_scaled = scaler.transform(test_deep_features_subset)

    # Get the predicted labels for each sample. first use the model
    test_predicted_labels = kmeans.predict(test_deep_features_subset_scaled)

    # and then the dictionaries
    test_predicted_labels = [cluster_to_label[value] for value in test_predicted_labels]
    pred_classes = [idx_to_class[value] for value in test_predicted_labels] # with names

    # get the true classes of the test set in numeric format
    test_true_labels = [class_to_idx[value] for value in test_labels_subset]

    ############################ MAKE CONFUSION MATRIX FOR TEST SET ##########################

    # Get unique class labels
    unique_labels_set = set(test_labels_subset)
    unique_labels = sorted(list(unique_labels_set))

    print(unique_labels)
    # Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Initialize confusion matrix with empty columns
    test_conf_matrix = [[0 for _ in unique_labels] for _ in unique_labels]

    # Fill in the confusion matrix
    for true_label, pred_class in zip(test_labels_subset, pred_classes):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_class]
        test_conf_matrix[true_index][pred_index] += 1

    # Save testing confusion matrix to CSV
    test_confusion_file_path = path_for_laptop / "test_confusion_matrix_kmeans.csv"

    with open(test_confusion_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["True Label"] + unique_labels)
        # Write data
        for i, row in enumerate(test_conf_matrix):
            writer.writerow([unique_labels[i]] + row)


    ############################ CALCULATE SOME METRICS ##########################

    kmeans_ari_score = adjusted_rand_score(test_labels_subset, pred_classes)
    print(f"\nAdjusted Rand Index: {kmeans_ari_score:.4f}")

    ########################### SAVE FILES ############################################

    # for the training set
    np.save(path_for_laptop / 'predicted_labels_kmeans.npy', pred_classes)
    np.save(path_for_laptop / 'true_labels_kmeans.npy', labels_subset)

    # for the testing set
    np.save(path_for_laptop / 'test_predicted_labels_kmeans.npy', test_predicted_labels)
    np.save(path_for_laptop / 'test_true_labels_kmeans.npy', test_labels_subset)




    ############################ PRINT FINAL MESSAGE  ##########################
    print('\nTo download the laptop folder to your computer, run this line in terminal: ')
    print('scp -r x_karga@berzelius.nsc.liu.se:/proj/berzelius-2023-48/ifcb/main_folder_karin/data/deep_feature_extraction/files_to_laptop /Users/forskningskarin/Documents/Courses/FID3214/Assignment_4/Code\n')