# k-NN Classifier on Wine Dataset

## Overview
This project implements the k-Nearest Neighbors (k-NN) classification algorithm on the Wine dataset. The goal is to classify different types of wines based on their chemical properties.

## Dataset
- **Source:** The Wine dataset is used for classification tasks and contains 13 numerical attributes.
- **Features:** Various chemical properties such as alcohol content, flavonoids, and magnesium levels.
- **Target Classes:** Three different types of wine labeled as 0, 1, and 2.

## Implementation
The implementation follows these steps:
1. Load the dataset from `wine.data`.
2. Preprocess the data by splitting it into training and test sets.
3. Normalize the data using `StandardScaler`.
4. Implement the k-NN algorithm from scratch.
5. Evaluate performance for different `k` values.
6. Visualize accuracy results.
7. Display the confusion matrix and classification report for the best `k`.

## Files
- `analysis.ipynb`: Jupyter Notebook containing the implementation, analysis, and visualization.
- `wine.data`: The dataset file.
- `wine.names`: Description of dataset attributes.

## How to Run
1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```
3. Run the notebook cells sequentially.

## Results
- The best `k` value is determined based on accuracy.
- The confusion matrix and classification report provide performance insights.
- The classifier achieves high accuracy in classifying wine samples.

## Conclusion
The k-NN classifier effectively classifies wine types using their chemical properties. The model's performance depends on the choice of `k`, which is tuned for optimal accuracy.

