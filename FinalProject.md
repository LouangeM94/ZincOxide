# AJL Kaggle Project - Break Through Tech AI

## Team Member
- **Louange Mizero** - [GitHub Profile](https://github.com/LouangeM94)

---

## Project Highlights
- Participated in the **AJL Kaggle competition** as part of the **Break Through Tech AI Program**.
- Developed a **computer vision model** for image classification using deep learning techniques.
- Implemented **data preprocessing, augmentation, and CNN-based architectures** for optimal performance.
- Achieved **97.3% training accuracy and 92% validation accuracy**.
- Conducted **exploratory data analysis (EDA)** to address class imbalances and ensure data consistency.
- Proposed **next steps**, including hyperparameter tuning and transfer learning for further improvements.

---

## Setup & Execution

### Requirements
To run this project, install the necessary dependencies:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

### Running the Model
1. **Clone the Repository**
   ```bash
   git clone https://github.com/LouangeM94/AJL-Kaggle-Project.git
   cd AJL-Kaggle-Project
   ```
2. **Prepare the Dataset**
   - Download the dataset from Kaggle and place it in the appropriate directory.
   - Ensure images and labels are correctly formatted.
3. **Train the Model**
   ```bash
   python train_model.py
   ```
4. **Evaluate Performance**
   ```bash
   python evaluate_model.py
   ```
5. **Generate Predictions**
   ```bash
   python predict.py --input test_images/
   ```

---

## Project Overview
### Kaggle Competition and Break Through Tech AI
This project is part of the **Break Through Tech AI Program**, which aims to empower underrepresented students in AI through real-world challenges. Our team is participating in the **AJL Kaggle competition**, leveraging machine learning techniques to solve a classification problem using image data.

### Objective of the Challenge
The goal is to develop a **computer vision model** that accurately classifies images based on their labels. Our approach includes **image preprocessing, data augmentation, and deep learning techniques** to optimize model performance.

### Real-World Significance
Image classification is a critical application in AI with wide-ranging real-world impacts, including:
- **Healthcare**: Detecting diseases in medical imaging
- **Security**: Identifying fraudulent activities through image recognition
- **Retail**: Automating product categorization for e-commerce platforms

---

## Data Exploration
### Dataset Description
The dataset consists of labeled images provided by Kaggle. Each image has an **md5hash** identifier and a corresponding **class label**. The training dataset includes image paths and class labels, while the test dataset consists of images without labels for model evaluation.

### Data Preprocessing Approaches
1. **Filepath Creation**: Converted `md5hash` values to proper image filenames.
2. **Label Encoding**: Transformed categorical labels into numerical values using `LabelEncoder`.
3. **Data Splitting**: Split the dataset into training (80%) and validation (20%) sets.
4. **Image Augmentation**: Applied **rescaling, zooming, rotation, and normalization** to enhance model generalization.

### Exploratory Data Analysis (EDA)
- **Class Distribution**: Addressed dataset imbalance through augmentation.
- **Image Resolution**: Verified consistency in image dimensions.
- **Sample Visualizations**: Displayed random images per class for better understanding.

---

## Model Development
### CNN Architecture
Our Convolutional Neural Network (CNN) includes:
- **Conv2D Layers**: Extract features from images.
- **MaxPooling**: Reduce dimensionality.
- **BatchNormalization**: Improve stability and convergence.
- **Dense Layers**: Learn complex patterns.
- **Dropout**: Prevent overfitting.

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy
- **Epochs**: 15
- **Batch Size**: 32

---

## Results & Key Findings
### Performance Metrics
- **Training Accuracy**: 97.3%
- **Validation Accuracy**: 92%
- **Loss Reduction**: Steady training loss decline indicates effective learning.

### Performance Visualizations
- **Accuracy Curve**: Demonstrates improvement over epochs.
- **Loss Curve**: Shows optimization progress.

---

## Impact Narrative
This project highlights our ability to apply **machine learning and deep learning techniques** to solve real-world AI challenges. By enhancing image classification accuracy, we contribute to industries such as **healthcare, security, and e-commerce**. The experience gained through this competition strengthens our technical skills and prepares us for advanced AI projects.

---

## Next Steps & Future Improvements
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and layer configurations.
- **Advanced Augmentation**: Experiment with techniques like cutout and mixup.
- **Transfer Learning**: Leverage pre-trained models for improved feature extraction.
- **Deploying the Model**: Implement an interactive web demo for real-time classification.
By continuing our efforts, we aim to push the boundaries of **computer vision** and enhance AI-driven image classification systems.


