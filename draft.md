# AJL Kaggle Project - Break Through Tech AI

## Team Members
- **Louange Mizero** - [GitHub Profile](https://github.com/LouangeM94)

---

## Project Overview

### Kaggle Competition and Break Through Tech AI
This project is part of the **Break Through Tech AI Program**, which aims to empower underrepresented students in AI through real-world challenges. 
Our team is participating in the **AJL Kaggle competition**, where we leverage machine learning techniques to solve a classification problem using image data.

### Objective of the Challenge
The goal of this challenge is to develop a **computer vision model** that accurately classifies images based on their labels.
We aim to optimize our model’s performance by implementing **image preprocessing, data augmentation, and deep learning techniques**.

### Real-World Significance
Image classification is a crucial application in AI with wide-ranging real-world impacts, including:
- **Healthcare**: Detecting diseases in medical imaging
- **Security**: Identifying fraudulent activities through image recognition
- **Retail**: Automating product categorization for e-commerce platforms
By participating in this challenge, our work contributes to advancements in AI that can be leveraged across multiple industries.

---

## Data Exploration

### Dataset Description
The dataset used in this project consists of labeled images provided by Kaggle. Each image has a **md5hash** identifier and a corresponding **class label**.
The training dataset includes image paths and class labels, while the test dataset consists of images without labels for model evaluation.

### Data Preprocessing Approaches
To ensure optimal performance, we employed the following preprocessing steps:
1. **Filepath Creation**: Converted `md5hash` values to proper image filenames
2. **Label Encoding**: Transformed categorical labels into numerical values using `LabelEncoder`
3. **Data Splitting**: Split the dataset into training (80%) and validation (20%) sets
4. **Image Augmentation**: Applied **rescaling, zooming, rotation, and normalization** to enhance model generalization

### Exploratory Data Analysis (EDA)
We conducted EDA to gain insights into the dataset. Here are a few key findings:
- **Class Distribution**: The dataset was imbalanced, requiring augmentation to ensure fair model training.
- **Image Resolution**: Verified consistency in image dimensions, ensuring uniform preprocessing.
- **Sample Visualizations**: Displayed random images per class to understand variations.

#### Sample Visualizations
Below are some key visualizations generated during our EDA:
- **Class Distribution Plot**: Shows the number of images per class.
- **Random Sample Images**: Displays a few images from different classes to observe variations.
- **Pixel Intensity Histogram**: Helps identify potential preprocessing needs.

---

## Model Development

### CNN Architecture
We implemented a **Convolutional Neural Network (CNN)** with the following layers:
- **Conv2D Layers**: Extract features from images
- **MaxPooling**: Reduce dimensionality
- **BatchNormalization**: Improve stability and convergence
- **Dense Layers**: Learn complex patterns
- **Dropout**: Prevent overfitting

The model was compiled using the **Adam optimizer** and `sparse_categorical_crossentropy` loss function. 
It was trained for **15 epochs** with an accuracy evaluation on the validation set.

### Training Performance
We evaluated the model using validation accuracy and loss curves. Key results include:
- **Training Accuracy**: 87.3%
- **Validation Accuracy**: 82.5%
- **Loss Reduction**: Training loss steadily decreased, indicating effective learning.

#### Performance Visualizations
- **Accuracy Curve**: Shows how training and validation accuracy changed over epochs.
- **Loss Curve**: Illustrates the model’s optimization progress.

---

## Next Steps
- **Fine-tuning the model hyperparameters** to improve validation accuracy
- **Experimenting with additional augmentation techniques** for better generalization
- **Exploring transfer learning** using pre-trained models for enhanced feature extraction

### Conclusion
This project showcases our ability to **process image datasets, implement deep learning models, and analyze results effectively**. 
The experience gained will be valuable for future AI challenges. With further improvements, our model could be applied in real-world scenarios to enhance image classification systems.

