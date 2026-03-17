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
* [Setup Instructions](#setup-instructions)
* [Code Structure](#code-structure)
* [Results and Evaluation](#results-and-evaluation)
* [Proposed Improvements](#proposed-improvements)
* [References](#references)


# Project Overview

This implementation is a submission for the final project of the class of "Pattern Recognition" (DUTH ECE: 9th Semester 2025-2026), for which students were tasked to develop two machine learning models. 

1. A system that detects faults in an industrial process that produces semiconductors. The input is a collection of sensor measurements in different stages of the production procedure and the model output a prediction of whether the samples present the desired normal operation or a faulty operation (binary classification).

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

    The submitted implemention involves the appropriate data preprocessing, including missing data handling, to train a Random Forest, the optimal parameters of which are achieved through hyperparameter searching and the proper validation.  

2. A system that solves the problem of image segmentation in a hyperspectral image. The goal is the creation of a map of land-cover (multiclass classification) for the research area using only the classification of the model. The input data are the higher dimensionality spectral vectors for each pixel of the image, which vector correspond to different wavelengths of the electromagnetic spectrum.

    ![Automatic ROI Patch](media/output_comparison.png)
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

### Task 1
The data provided for this task are the training set file `Training_data_manifacturing.csv` and the test set file `Test_data_manifacturing.csv`. 

> **Note:** There is a strong possibility that the provided data is a modified part of a larger publicly available dataset, but its exact origin was not disclosed. Consequently, there is no direct attribution or link provided here. The inclusion of this data is strictly for educational purposes and code reproducibility.


* **Data Description:** 

  The `Training_data_manifacturing.csv` training set consists of 1254 samples with each having 474 features. The samples are organized in lines, the features in columns and in the final column are stored the class labels of each sample, where '0' signals normal and '1' faulty operation. There is huge class imbalance (0->93.3%, 1->6.7%) and the 5.53% of all values is missing.
  
  The `Test_data_manifacturing.csv` test set is made of 313 samples with the same ration of mising values, but there is no class column, meaning that the true labels are hidden.


### Task 2

The data provided for this task is the `HyperspectralTask.mat` which is a hyperspectral image of **Pavia University** provided by Prof. Paolo Gamba[^1]. The file used is a modified version of the original as it the image is cropped and the ground truth partial.


* **Original Source Link:** [Pavia University Scene](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)
* **Modified Data Description:** The `HyperspectralTask.mat` is hyperspectral image formatted as *MATLAB v7.3 (HDF5)*. The file contains a 3-dimensional data cube of 610px height, 340px width and 103 spectral bands, but also the ground truth of 610 height and 340 width. The classes correspond to the labels '1' through '9', but not all pixels are labeled with the majority of them belonging to unknown class '0'.
* **Data Preprocessing:** The `Final_Project_PR_58545.ipynb` notebook already takes care of it, but it must be mentioned that the data stored in `HyperspectralTask.mat` need to be transposed so they are arranged as ***(height x width x spectral bands)*** and not as the original ***(spectral bands x width x height)***.


# Setup Instructions

1. **Load the Notebook:** Upload and open the `Final_Project_PR_58545.ipynb` file in Google Colab.
2. **Load the Data:** Go the data directory, download the `Final_Project_data.zip` file. No need to extract the files, just upload it directly into Colab's temporary session storage using the ***Files*** section on the left sidebar and run the first code cell of `Final_Project_PR_58545.ipynb`.
3. **Force-install Specific Package Versions:** Because cloud environments update their software frequently, newer versions of libraries might cause compatibility issues. If you encounter any unexpected errors while running the cells, please force-install the specific package versions listed in the [Python Packages Used](#codes-and-resources-used) section. To do that create a new code cell and run, for example, the command `!pip install scikit-learn==1.5.0`.
4. **Complete Setup:** Run the two last code cells of the *Setup* section of `Final_Project_PR_58545.ipynb` to import the utilized libraries and the define the function that sets the random seeds for reproducibility.


# Code Structure



* **Part 1: Data Analysis**

  In this sector the `vid.avi` video file and `ground_truth.txt` ground truth file of subject 45 are analyzed in order to print useful for the user information about them. Useful information includes sample frame, total duration and frames, waveform of ground truth PPG and Heart Rate, etc.

  ![Sample frame and Results](media/subject45_frame.png)
  
  *Fig. 2: Sample video frame of subject 45 of UBFC-rPPG Dataset[^1].*

* **Part 2: Heart Rate Estimation**

  In this part, the algorithm for extracting the subject's Heart Rate is applied, following the below steps. It must be noted that the algorithm was developed using the methodology proposed by Berggren & Berggren (2019)[^2] as a primary reference.

  1. **ROI Definition**: The proposed algorithm does not use the whole video frame but a patch of subject 45 skin on their forehead. In this implementation two ways of defining the Region Of Interest (ROI) are tested: **a) Manual Definition**, where a stationary pixel area is predefined as the ROI for the whole video, and **b) Definition through Face Tracking**, where a patch of constant area is chosen automatically in every frame of the video using the *Viola and Jones*[^3] face detection algorithm which is based on the principle of Haar-like features.

      ![Manual ROI Patch](media/manual_roi.png)
      *Fig. 3: Manual ROI definition for a single frame of subject 45 video.*

      ![Automatic ROI Patch](media/face_detection_roi.png)
      *Fig. 4: ROI definition through face detection for a single frame of subject 45 video.*

      > **Note:** The next steps of the algorithm are applied to both Manual and Automatic ROI. However, since the results are similar and Automatic ROI is the objectively better and smarter implementation, for the following steps only the results of this ROI are presented.

  2. **Spatial Averaging**: For the ROI of every video frame, the mean pixel intensity of a single color channel—in this case the green channel as it is more suitable for rPPG[^4]—is calculated. These sequential averages are then plotted over time to construct a raw waveform, which serves as the basis for extracting the heart rate signal.

      ![Spatial Averaging Result](media/spatial_averaging.png)
      *Fig. 5: Resulting waveform of spatially averaging the green channel of the ROI of every frame.*

  3. **Normalization**: Z-score normalization is applied to the raw waveform.

      ![Normalized Mean Intensity](media/normalized_signal.png)
      *Fig. 6: Normalized waveform of mean channel intensity.*
      
  4. **Bandpass Filtering**: The normalized waveform is bandpass filtered using a first order Butterworth filter of 0.75Hz-3.5Hz in order to isolate the possible cardial frequencies (45bpm-210bpm).

      ![Denoised Mean Intensity](media/denoised_signal.png)
      *Fig. 7: Bandpass filtered (0.75Hz-3.5Hz 1st order Butterworth filter) waveform of mean channel intensity.*

  5. **STFT**: To extract the heart rate estimation, Short-Time Fourier Transform is applied to the filtered waveform.

      The parameters of the STFT are:

      ```python
      fs = 29.951  # Sampling frequency ~= 30Hz
      window_size = 215 
      overlap_ratio = 0.5
      ``` 

      The maximum frequency of each time window corresponds to the HR prediction.

      ![STFT Spectogram](media/spectogram.png)
      *Fig. 8: Spectogram of the STFT of the filtered waveform.*


* **Part 3: Complementary Analysis**

  In this part some extra analysis is performed to resolve 2 issues:

  1. **Deviation between estimation and ground truth**: In order to evaluate the effectiveness of the method, the differences between the estimation and target HR are examined by calculating the frequency (instantaneous HR from peak frequency for the ground truth and FFT for the estimation) in the problem timeframes.

  2. **Algorithm weaknesses**: The method is applied to the subject 11 video where there are some periodic lighting changes, in order to uncover the algorithm's inaccuracies.


# Results and Evaluation

The application of the proposed methodology to the subject 45 video leads to the predictions of *Fig. 9*.

![Estimation vs Ground Truth comparison](media/hr_comparison.png)
*Fig. 9: Comparison of estimated heart rate and ground truth.*

The extra analysis of [*Part 3*](#code-structure) shows that the deviation at 53s-58s is not due to an inaccuracy of the method, but due to an issue with the ground truth. As for the time shift between estimation and ground truth, it is not a quirk, but it derives from the fact that the proposed algorithm offers offline predictions, while the oximeter gives online, and therefore, delayed readings.

However, the somewhat big size of STFT window leads to estimations that are able to follow the general changes of the heart rate, but fail to capture the small "local" fluctuations. It should also be mentioned that the values of the STFT parameters are a product of fine tuning instead of being chosen automatically. Moreover, this method fails during significant light changes and subject movements.


# Proposed Improvements

Nowadays there are many advanced rPPG methods that are way more efficient and accurate and possess way less weaknesses than the proposed method. However, should someone want to retain the proposed methodology, but improve its performance, there are some improvements that could be made:

* Replace STFT with a more effective method of frequency calculation (eg. Continuous Wavelet Transform - CWT).

* Implement a mechanism that chooses automatically the optimal parameters for frequency calculation.

* Modify the method of measuring the channel intensity so it is more robust to motional and lighting changes.


# References

[^1]: P. Gamba, "Pavia University Hyperspectral Dataset", Telecommunications and Remote Sensing Laboratory, Pavia University, Italy, 2001. Available: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

[^2]: Berggrem, A., Berggrem J. (2019) Non-contact measurement of heart rate using a camera, [Master's thesis, Lund University]. lup.lub.lu.se. [http://lup.lub.lu.se/student-papers/record/8972235](http://lup.lub.lu.se/student-papers/record/8972235)

[^3]: P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, Kauai, HI, USA, 2001, pp. I-I, doi: 10.1109/CVPR.2001.990517.

[^4]: Verkruysse, Wim & Svaasand, Lars & Nelson, John. (2008). Remote plethysmographic imaging using ambient light.. Optics Express. 16. 21434-21445. 10.1364/OE.16.021434. 