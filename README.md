# Pattern Recognition Final Project


![GitHub last commit](https://img.shields.io/github/last-commit/iliaxant/Pattern_Recognition_Final_Project?path=Final_Project_PR_58545.ipynb)
![Python Version](https://img.shields.io/badge/Python-3.12-orange.svg?logo=python&logoColor=white) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iliaxant/Pattern_Recognition_Final_Project/blob/main/Final_Project_PR_58545.ipynb)


> A set of Machine Learning models developed for Pattern Recognition tasks, featuring industrial process fault detection (binary classification) and hyperspectral image (multiclass) classification.


# Table of Contents

* [Project Overview](#project-overview)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)
  * [Codes and Resources Used](#codes-and-resources-used)
  * [Python Packages Used](#python-packages-used)
* [Data](#data)
  * [Task 1](#task-1-1)
  * [Task 2](#task-2-1)
* [Setup Instructions](#setup-instructions)
* [Task 1](#task-1-2)
  * [Code Structure](#code-structure)
  * [Results and Evaluation](#results-and-evaluation)
* [Task 2](#task-2-2)
  * [Code Structure](#code-structure)
  * [Results and Evaluation](#results-and-evaluation)
* [Proposed Improvements](#proposed-improvements)
* [References](#references)


# Project Overview

This implementation is a submission for the final project of the class of "Pattern Recognition" (DUTH ECE: 9th Semester 2025-2026), for which students were tasked to develop two machine learning models. 

### Task 1

A system that detects faults in an industrial process that produces semiconductors. The input is a collection of sensor measurements in different stages of the production procedure and the model output a prediction of whether the samples present the desired normal operation or a faulty operation (binary classification).

```
    Probability
0       0.189140
1       0.202409
2       0.076705
3       0.133381
4       0.075945
..           ...
308     0.024652
309     0.026147
310     0.086703
311     0.046149
312     0.046005
```
*Fig. 1: The model output predictions that correspond to the probability of the test set samples being faulty.*

The submitted implemention involves the appropriate data preprocessing, including missing data handling, to train a Random Forest classifier, the optimal parameters of which are achieved through hyperparameter searching and the proper validation.  

### Task 2

A system that solves the problem of image segmentation in a hyperspectral image. The goal is the creation of a map of land-cover (multiclass classification) of the research area using only the classification of the model. The input data are the higher dimensionality spectral vectors for each pixel of the image, which vector correspond to different wavelengths of the electromagnetic spectrum.

![Output Comparison](media/output_comparison.png)
*Fig. 2: The (approximated) research area (left) and the output pixel map (right).*

The developed solution performs data preprocessing, featuring average filtering for utilizing the data morphology, trains and validates a Support Vector Machine with RBF kernel after a proper hyperparameter optimization and postprocesses the final pixel map by applying a median filter on pixels of weak predictions.

The test set outputs of both models were evaluated by the class professor and all student submissions were ranked based on performance. The present implementation of the second task has achieved the **3rd highest F1-score** of the 2025-2026 class, while the first model has achieved a high place in the ranking but not a podium spot.


# Project Structure

The repository is structured as below:
```bash
├── data
│   └── Final_Project_data.zip
├── media
├── predictions
│   ├── test_predictions_task1_58545.csv
│   └── test_predictions_task2_58545.csv
├── 58545_.pdf
├── Final_Project_PR_58545.ipynb
└── README.md
```
* `data`: Directory containing the `Final_Project_data.zip` which possess the data files: `Training_data_manifacturing.csv` and `Test_data_manifacturing.csv` for the first problem and `HyperspectralTask.mat` for the second.
* `media`: Directory containing pictures used in README.md.
* `predictions`: Directory containing the two submissions for the evaluation and ranking of the implementations.
* `58545_.pdf`: A short presentation *(in greek)* highlighting the methodology followed and the achieved validation performance of both systems.
* `Final_Project_PR_58545.ipynb`: The heart of the project. A Jupyter Notebook containing the step-by-step implementation of both machine learning models *(text cells in greek)*. 


# Dependencies

## Codes and Resources Used
* **Editor Used:** Google Colab / Jupyter Notebook
* **Python Version:** 3.12 (Google Colab Default Runtime at the time of the project)

## Python Packages Used
All the necessary dependencies needed for the reproduction of the project are categorized as follows:

> **Note:** Since this project is developed entirely within Google Colab, the package versions listed below correspond to the default Colab environment at the time of the last code update (Febuary 2026).

* **General Purpose & Utilities:**
  * `zipfile` *(Built-in with Python 3.12)*: For extracting dataset archives.
  * `random` *(Built-in with Python 3.12)*: For generating random numbers and setting random states for reproducibility.
  * `h5py` *(v3.11.x)*: For reading and interacting with HDF5 formatted data.

* **Data Manipulation & Analysis:**
  * `numpy` *(v2.1.x)*: For numerical operations, matrix handling and array manipulation.
  * `pandas` *(v2.2.x)*: For data structuring, handling tabular data and manipulation.

* **Machine Learning & Pattern Recognition:**
  * `scikit-learn` *(v1.5.x)*: The core library used for the machine learning pipeline. Specifically utilized for:
    * **Preprocessing:** `StandardScaler`, `SimpleImputer`
    * **Dimensionality Reduction:** `PCA`
    * **Model Selection & Tuning:** `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, `Pipeline`
    * **Classifiers:** `RandomForestClassifier`, `SVC`
    * **Metrics:** `accuracy_score`, `confusion_matrix`, `classification_report`, `roc_auc_score`, `f1_score`, etc.

* **Scientific Computing & Signal Processing:**
  * `scipy` *(v1.14.x)*: Utilized for defining statistical distributions for hyperparameter tuning and applying spatial filters to data.

* **Data Visualization:**
  * `matplotlib` *(v3.9.x)*: Used for foundational plotting and custom color mapping.
  * `seaborn` *(v0.13.x)*: For generating custom color palettes.


# Data

## Task 1
The data provided for this task are the training set file `Training_data_manifacturing.csv` and the test set file `Test_data_manifacturing.csv`. 

> **Note:** There is a strong possibility that the provided data is a modified part of a larger publicly available dataset, but its exact origin was not disclosed. Consequently, there is no direct attribution or link provided here. The inclusion of this data is strictly for educational purposes and code reproducibility.


* **Data Description:** 

  The `Training_data_manifacturing.csv` training set consists of 1254 samples with each having 474 features. The samples are organized in lines, the features in columns and in the final column are stored the class labels of each sample, where '0' signals normal and '1' faulty operation. There is huge class imbalance (0->93.3%, 1->6.7%) and the 5.53% of all values is missing.
  
  The `Test_data_manifacturing.csv` test set is made of 313 samples with the same ratio of mising values, but there is no class column, meaning that the true labels are hidden.


## Task 2

The data provided for this task is the `HyperspectralTask.mat` which is a hyperspectral image of **Pavia University** provided by Prof. Paolo Gamba[^1]. The file used is a modified version of the original as it the image is cropped and the ground truth partial.


* **Original Source Link:** [Pavia University Scene](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)
* **Modified Data Description:** The `HyperspectralTask.mat` is hyperspectral image formatted as *MATLAB v7.3 (HDF5)*. The file contains a 3-dimensional data cube of 610px height, 340px width and 103 spectral bands, but also the ground truth of 610 height and 340 width. The classes correspond to the labels '1' through '9', but not all pixels are labeled with the majority of them belonging to unknown class '0'. The classes of the labeled pixels are heavily imblanced with the most common class reaching 53.66% and least common 1.27%.
* **Data Preprocessing:** The `Final_Project_PR_58545.ipynb` notebook already takes care of it, but it must be mentioned that the data stored in `HyperspectralTask.mat` need to be transposed so they are arranged as ***(height x width x spectral bands)*** and not as the original ***(spectral bands x width x height)***.


# Setup Instructions

1. **Load the Notebook:** Upload and open the `Final_Project_PR_58545.ipynb` file in Google Colab.
2. **Load the Data:** Go the data directory, download the `Final_Project_data.zip` file. No need to extract the files, just upload it directly into Colab's temporary session storage using the ***Files*** section on the left sidebar and run the first code cell of `Final_Project_PR_58545.ipynb`.
3. **Force-install Specific Package Versions:** Because cloud environments update their software frequently, newer versions of libraries might cause compatibility issues. If you encounter any unexpected errors while running the cells, please force-install the specific package versions listed in the [Python Packages Used](#codes-and-resources-used) section. To do that create a new code cell and run, for example, the command `!pip install scikit-learn==1.5.0`.
4. **Complete Setup:** Run the two last code cells of the *Setup* section of `Final_Project_PR_58545.ipynb` to import the utilized libraries and the define the function that sets the random seeds for reproducibility.


# Task 1

## Code Structure

### Part A: Data Analysis

In this sector the contents of `Training_data_manifacturing.csv` and `Test_data_manifacturing.csv` are analyzed in order to print useful for the user information about the data. Useful information involves percentages of each class in the training set and the ratio of missing values for each feature.


### Part B: Data Preprocessing

The applied methods of data processing involve

1. **Removal of (quasi-) constant features:** Features with mostly constant values (specifically with >=99% consistency) are removed in order to reduce redundancy.

    ```
    Constant / Quasi-Constant (>= 99.0% same value) Feature list:
    ['Feat 68', 'Feat 188', 'Feat 191', 'Feat 287', 'Feat 292', 'Feat 388']

    Removing 6 Constant / Quasi-Constant Features...
    ```
    *Fig. 3: Result/output of (quasi-) constant feature removal.*


2. **Removal of features with excessive missing values:** Features with missing value ratio of 80% and over are removed as a method of dimensionality reduction, provided the data are not **Missing Not At Random (MNAR)**. In this project features with data MNAR are identified when the difference of the amount of missing value between classes is greater or equal to 10%.

    ```
    Missing >=80.0% of values Feature list:
    ['Feat 79', 'Feat 148', 'Feat 149', 'Feat 202', 'Feat 247', 'Feat 248', 'Feat 303', 'Feat 401']

    ---------- MNAR Check ----------

    Feat 79 : Missing Rate Class 0(-) = 85.56% | Class 1(+) = 90.48% | Diff =   4.92%
    Feat 148: Missing Rate Class 0(-) = 91.79% | Class 1(+) = 94.05% | Diff =   2.25%
    Feat 149: Missing Rate Class 0(-) = 91.79% | Class 1(+) = 94.05% | Diff =   2.25%
    Feat 202: Missing Rate Class 0(-) = 85.56% | Class 1(+) = 90.48% | Diff =   4.92%
    Feat 247: Missing Rate Class 0(-) = 91.79% | Class 1(+) = 94.05% | Diff =   2.25%
    Feat 248: Missing Rate Class 0(-) = 91.79% | Class 1(+) = 94.05% | Diff =   2.25%
    Feat 303: Missing Rate Class 0(-) = 85.56% | Class 1(+) = 90.48% | Diff =   4.92%
    Feat 401: Missing Rate Class 0(-) = 85.56% | Class 1(+) = 90.48% | Diff =   4.92%

    Summary: Possible MNAR (Diff >= 10.0%) -> 
    []

    --------------------------------

    Removing 8 features:
    ['Feat 79', 'Feat 148', 'Feat 149', 'Feat 202', 'Feat 247', 'Feat 248', 'Feat 303', 'Feat 401']
    ```

    *Fig. 4: Result/output of proccess of removing features with excessive missing values.*

3. **Missing Value Handling**: Τhe NaN cells of the remaining features are filled using  *SimpleImputer()* of sklearn, meaning with simple replacement with the same value in all cells of a feature. In this implementation, the way the replacement value is calculated is a hyperparameter.
    

### Part C: Data Splitting

The training data is split into 5 folds so K-Fold Crossover Validation is possible. The splitting occurs after shuffling the samples and it is stratified sο the ratio between the two classes within each fold is maintained.


### Part D: Choosing a classifier

Although there is no code for this part, the classifier choice has to be mentioned. The task is solved using a Random Forest classifier with bootstrapping enabled and with rest of its parameters being hyperparameters to be optimized.


### Part E: Hyperparameter optimization

In this section the model is trained multiple times to find the optimal hyperparameters. The optimization occurs in two parts:

1. **Hyperparameter searching**: The hyperparameter space is searched to find the optimal set. The search is conducted using Random Search with Crossover Validation using the splits of *Part C* and having as a metric the Area under ROC curve. The searching pipeline consists of filling the missing data cells as per *Part B.3* and then training and validating, both individually for each fold. The Random Search has been executed multiple times with different distributions, but each search has 100 iterations and the searched parameters (with the distributions of the final search) are:

    ```python
    'imputer__strategy': ['median', 'most_frequent'],
    'clf__n_estimators': randint(200, 1200),
    'clf__max_depth': randint(10, 25),
    'clf__min_samples_leaf': randint(2, 15),
    'clf__min_samples_split': randint(2, 20),
    'clf__max_features': uniform(0.01, 0.97),
    'clf__class_weight': ['balanced', 'balanced_subsample']
    ```

2. **Fine Tuning**: After a satisfying number of searches, the same, as with the search, pipeline is used to further train and validate the classifier, but only for a specific set of hyperparameters at a time in order to fine tune the model, utilizing the information obtained during the searching. In this part, more detailed information about the training and validation are printed, including ROC AUC and PR AUC, and an optimal threshold for the predictions is chosen by maximizing the F1-score. The deciding factor when choosing the best set of hyperparameters is whether it maximizes **ROC AUC**.


### Part F: Model retraining and testing

After settling on the optimal hyperparameters, the classifier is retrained on all the training data (meaning there is no validation) and then tested on the test set to produce the predictions neccesary for the submitted `test_predictions_task1_58545.csv` file as per instructed. 


## Results and Evaluation

The optimal classifier discovered during the optimization of *Part E* is the one with the below set of hyperparameters:

```python
imp_strategy = 'most_frequent'
rf_n_classifiers = 800
rf_max_depth = 22
rf_min_samples_leaf = 5
rf_min_samples_split = 15
rf_max_features = 0.3009
rf_class_weight = 'balanced'
```
Training and validating model using the apove parameters leads to the following results and metrics:

```
--- Fold 1 ---
ROC AUC: 0.7076
PR AUC: 0.2332
Optimal (using F1-score) Threshold: 0.1750
F1 Score: 30.19%
Precision: 22.22%  |  Recall: 47.06%
Confusion Matrix:
 [[206  28]
 [  9   8]]

--- Fold 2 ---
ROC AUC: 0.8436
PR AUC: 0.4074
Optimal (using F1-score) Threshold: 0.1904
F1 Score: 40.00%
Precision: 34.78%  |  Recall: 47.06%
Confusion Matrix:
 [[219  15]
 [  9   8]]

--- Fold 3 ---
ROC AUC: 0.8049
PR AUC: 0.2557
Optimal (using F1-score) Threshold: 0.1611
F1 Score: 37.29%
Precision: 26.19%  |  Recall: 64.71%
Confusion Matrix:
 [[203  31]
 [  6  11]]

--- Fold 4 ---
ROC AUC: 0.7474
PR AUC: 0.1838
Optimal (using F1-score) Threshold: 0.1993
F1 Score: 33.33%
Precision: 25.81%  |  Recall: 47.06%
Confusion Matrix:
 [[211  23]
 [  9   8]]

--- Fold 5 ---
ROC AUC: 0.7895
PR AUC: 0.1883
Optimal (using F1-score) Threshold: 0.1539
F1 Score: 31.43%
Precision: 20.37%  |  Recall: 68.75%
Confusion Matrix:
 [[191  43]
 [  5  11]]

=============== Training Summary ===============

Average ROC AUC (Random Classifier = 0.5000): 0.7786  (std = 0.0470)
Average PR AUC (Baseline = 0.0670): 0.2537  (std = 0.0815)
Average Optimal Threshold: 0.1760  (std = 0.0171)

--- After Optimal Threshold ---
Average Precision: 25.87%  (std = 4.96%)
Average Recall:    54.93%  (std = 9.72%)
Average F1 Score:  34.45%  (std = 0.04%)

================================================
```

Part of the predictions produced during the final retraining and testing of *Part F* can be seen in *Fig. 1*.

The deciding factor for the ranking of the students' implementations was the Area under ROC curve. Although the present submission has not a achieved a podium position, the model tested higher than 0.8 ROC AUC, exceeding the estimated ROC AUC of 0.7786. 


# Task 2

## Code Structure

### Part A: Data Analysis

In this sector the contents of `HyperspectralTask.mat` are converted into an appropriate format and then are analyzed in order to print useful for the user information about the data. Useful information involves percentages of each class in the training set, sample channel and a plot of the per class mean specttral signature.

![Data Inspection](media/map_inspection.png)
*Fig. 5: a) Greyscale image of a random channel of the hyperspectral image, b) Ground Truth of Pixel map.*


### Part B: Data Preprocessing

The applied methods of data processing involve:

1. **Data augmentation through spatial averaging:** To each pixel of each channel (feature) of the original image, a 5x5 averaging (low pass) filter with mirroring for dealing with the pixel of border is applied. This new channel images act as extra features, thus doubling the dimensions of the problem, but now including the morphology / spatial correlation of the data. 

    ```
    Spatial averaging with a (5x5) filter...

    Original Data shape: (610, 340, 103)
    Augmented Data shape (Spectral Info + Spatial Mean): (610, 340, 206)

    Original Num of features: 103
    Num of features after augmentation: 206
    ```
    *Fig. 6: Output messages of feature augmentation through spatial averaging.*


2. **Normalization:** Z-score normalization is applied to the whole and augmented dataset. The mean and std are calculated using the whole dataset and not just the training data, as this problem belongs to the category of transductive machine learning.


### Part C: Data Splitting

The training data is split into 5 folds so K-Fold Crossover Validation is possible. The splitting occurs after shuffling the samples and it is stratified sο the ratio between the two classes within each fold is maintained.


### Part D: Choosing a classifier

Although there is no code for this part, the classifier choice has to be mentioned. The task is solved using a SVM with RBF kernel as, according to relavant papers[^2] [^3], it is an acceptable and efficient solution to the problem of multiclass classification in hyperspectral images. The `C` and `gamma` parameters of the classifier are the to be optimized hyperparameters of the model.


### Part E: Hyperparameter optimization

In this section the model is trained multiple times to find the optimal hyperparameters. The optimization occurs in two parts:

1. **Hyperparameter searching**: The hyperparameter space is searched to find the optimal set. The search is conducted using Grid Search with Crossover Validation that uses the splits of *Part C*, with as a metric the macro F1-score. The Grid Search algorithm has been executed multiple times for different grids, since the chosen classifier has a very small training time and the hyperparameters are only two in number as shown below in the final grid searched:

    ```python
    'clf__C': [500, 600, 800, 1000, 2000, 4000, 5000],
    'clf__gamma': [0.005, 0.006, 0.008, 0.01, 0.02, 0.04, 0.05]
    ```

2. **Fine Tuning**: After a satisfying number of searches, the classifier is further trained and validated, but only for a specific set of hyperparameters at a time in order to fine tune the model, utilizing the information obtained during the searching. In this part, more detailed information about the training and validation are printed, including accuracy, macro F1 and weighted F1 score. The deciding factor when choosing the best set of hyperparameters is whether it maximizes the **macro F1 score**.


### Part F: Model retraining and testing

After settling on the optimal hyperparameters, the classifier is retrained on all the training data (meaning there is no validation) and then tested on the whole dataset to produce a pixel map of predictions.

### Part G: Post-processing

In this section two post processes are tested in order to find the one that (hopefully) increases the accuracy of the predictions:

1. **Median Filtering**: A 3x3 median filter with mirroring for border pixels is applied to all pixels of the model's pixel map as a way to smooth out the predictions. 

2. **"Smart" Median Filter**: In order to avoid oversmoothing the pixel map of prediction, "smart" median filtering is applied to the model's pixel map. "Smart" means that instead of applying the filter to all pixels of the map, the filter is applied only to those prediction the model is not confident in. The threshold for whether the pixel is filtered is chosen to be equal to 2.75, which is manually set based on the distribution of confidence scores.

    ![Confidence scores](media/confidence_thresh.png)
    *Fig. 7: The distribution of prediction confidence scores the chosen threshold.*

### Part H: Finalizing predictions

Based on visual inspection and the worsening of predictions for the labeled pixels, the final pixel map of predictions is chosen to create the `test_predictions_task2_58545.csv` submission file as per instructed. The pixel map chosen here is the one after "smart" median filtering.


## Results and Evaluation

The optimal classifier discovered during the optimization of *Part E* is the one with the below set of hyperparameters:

```python
svm_C = 1000
svm_gamma = 0.02
```
Training and validating model using the apove parameters leads to the following results:

```
=============== Training Summary ===============

Average Accuracy:    0.9970  (std = 0.0011)
Average Macro F1:    0.9931  (std = 0.0025)
Average Weighted F1: 0.9970  (std = 0.0011)

================================================
```

The pixel map produced after the final retraining and testing of *Part F* can be seen in *Fig. 8.a)*.

The resulting map after the post-process of median filtering can be seen in *Fig. 8.c)*, while the map after "smart" median filtering in *Fig. 8.d)*

![Resulting Pixel Maps](media/pixelmaps.png)
*Fig. 8: The resulting pixel maps. From left to right: a) before post-processing, b) the approximated real image of the scene, c) after median filtering, d) after "smart" median filtering.*

As seen in *Fig. 8* and the below *Fig. 9*, "smart" median filter produces a not oversmoothed pixel map with less worsening of correct predictions, that is why it is considered the most appropriate post-process for this situation.

```
Total pixels filtered: 4628/207400 -> 2.23%/100%    # "Smart Median Filtering" 

Macro F1 before Post-Processing:         1.000000
Macro F1 after Median Filtering:         0.999674
Macro F1 after "Smart" Median Filtering: 0.999744
```
*Fig. 9: The worsening of F1 score before and after post-processing.*

The deciding factor for the ranking of the students' implementations was the **balanced accuracy**. Although the present submission has tested way lower than the estimated performance during validation and has suffered serious overfitting, it has achieved the 3rd highest scoring of all sumbitted implementations.


# Proposed Improvements

The nature of pattern recognition and machine learning problems is such that mostly there are not a set changes that will surely improve the performance of the model. So proposing improvements is somewhat futile as those upgrades can only be found by testing diffent parameters, classifiers, architectures and pipelines in general. However there are a couple of things to consider or change in order to improve the model, specifically for **Task 2** .

* Introducing to the training of the model more methods of overfitting prevention.

* Developing a process that chooses the optimal threhold for "smart" median filtering based on somewhat objective data and not on visual inspection


# References

[^1]: P. Gamba, "Pavia University Hyperspectral Dataset", Telecommunications and Remote Sensing Laboratory, Pavia University, Italy, 2001. Available: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

[^2]: F. Melgani and L. Bruzzone, "Classification of hyperspectral remote sensing images with support vector machines," in IEEE Transactions on Geoscience and Remote Sensing, vol. 42, no. 8, pp. 1778-1790, Aug. 2004, doi: 10.1109/TGRS.2004.831865.

[^3]: P. Ghamisi et al., "Advances in Hyperspectral Image and Signal Processing: A Comprehensive Overview of the State of the Art," in IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 4, pp. 37-78, Dec. 2017, doi: 10.1109/MGRS.2017.2762087.