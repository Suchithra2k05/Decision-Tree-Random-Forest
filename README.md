# Decision-Tree-Random-Forest

🌳 Decision Tree vs Random Forest Classifier
This project compares the performance of a Decision Tree Classifier and a Random Forest Classifier using the Iris Dataset. It also demonstrates hyperparameter tuning with GridSearchCV to find the best model configurations.

📌 About the Models
Decision Tree Classifier
A tree-structured model that splits data based on feature values.

Easy to interpret but can overfit on small datasets.

Works well for small to medium-sized datasets.

Random Forest Classifier
An ensemble method combining multiple decision trees.

Reduces overfitting by averaging results from many trees.

Usually provides better accuracy and generalization.

🎯 Project Objectives
Load and explore the Iris dataset.

Train a Decision Tree model and tune parameters.

Train a Random Forest model and tune parameters.

Compare accuracy and classification reports.

📂 Workflow
Load Dataset

Iris dataset from sklearn.datasets.

Split Data

70% training, 30% testing.

Train Decision Tree

Tune max_depth and min_samples_split.

Train Random Forest

Tune n_estimators, max_depth, and min_samples_split.

Evaluate Models

Compare accuracy scores and classification reports.

📊 Sample Output
yaml
Copy
Edit
Decision Tree Accuracy: 0.9777
Random Forest Accuracy: 0.9777

Classification Report - Decision Tree:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        19
           1       1.00      0.94      0.97        18
           2       0.94      1.00      0.97        13

Classification Report - Random Forest:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        19
           1       1.00      0.94      0.97        18
           2       0.94      1.00      0.97        13

Best Decision Tree Parameters: {'max_depth': 4, 'min_samples_split': 2}
Best Random Forest Parameters: {'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 100}
🏢 Internship Context
This project was created as Task 2 during my internship with Code n Career, focusing on comparing single-model vs ensemble-model performance for classification tasks.

💡 Key Learnings
Decision Trees are easy to interpret but prone to overfitting.

Random Forests generally provide better accuracy and robustness.

Hyperparameter tuning is essential to optimize model performance.

📌 How to Run
bash
Copy
Edit
# Clone the repository
git clone https://github.com/yourusername/decision-tree-vs-random-forest.git

# Navigate to the folder
cd decision-tree-vs-random-forest

# Install dependencies
pip install pandas scikit-learn

# Run the script
python decision_tree_vs_rf.py

🔮 Next Steps

Try other datasets like Wine or Breast Cancer for comparison.
Experiment with feature scaling and feature selection.
Visualize decision boundaries for better understanding.

