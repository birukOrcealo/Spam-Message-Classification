
![Static Badge](https://img.shields.io/badge/python-%233776AB?style=flat-square&logo=python-%233776AB&label=VERSION%203.11)  
![Static Badge](https://img.shields.io/badge/pandas-%23150458?style=flat&logo=pandas&label=VERSION%202.2.1)
![Static Badge](https://img.shields.io/badge/scikitlearn-%F7931E?style=flat&logo=scikitlearn&logoColor=black&labelColor=white&color=green)
## Spam Message Classification
This Python script performs spam message classification using machine learning techniques. It utilizes scikit-learn for building a text processing and classification pipeline, specifically employing TF-IDF vectorization and logistic regression.

## Requirements
   - Python 3.x
   -  pandas
   -  matplotlib
   - scikit-learn
## Usage
1. Ensure you have Python installed on your system.
2. Install the required libraries using pip:

```bash 
pip install pandas matplotlib scikit-learn

```
3. Download the 'spam.csv' dataset or specify the correct path to the dataset in the script.
4. Run the script:

## Description
- **spam_classification.py:** This script loads the spam dataset, preprocesses the data, splits it into training and testing sets, creates a text processing and classification pipeline, fits the pipeline to the training data, predicts labels for the testing data, and evaluates the performance using accuracy, precision, recall, and ROC AUC score. It also generates and displays a confusion matrix.

- **Libraries Used:**

    - pandas: For data manipulation and analysis.
    - matplotlib: For data visualization.
    - scikit-learn: For machine learning algorithms and utilities.
## Notes
- Ensure the path to the 'spam.csv' dataset is correct.
- The script converts the labels to binary format ('ham': 0, 'spam': 1).
- The size of the testing data is set to 20% of the total dataset, and the random  state is fixed for reproducibility.
- The classification pipeline consists of TF-IDF vectorization followed by logistic regression.
- The evaluation metrics used are accuracy, precision, recall, and ROC AUC score.
- The confusion matrix is plotted using the ConfusionMatrixDisplay class from -scikit-learn and displayed using matplotlib.
