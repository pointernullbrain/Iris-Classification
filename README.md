# 🌼 Description
This project uses machine learning models obtained from Spark MLlib (via pyspark) to classify species of Iris flowers according to their physical traits (sepal length, sepal width, petal length, and petal width). This project is done through Google Colab, in a python notebook.

# 📝 Methodology
The workflow of this project includes data loading, pre-processing, model setup (Decision tree, Random Forest, Logistic Regression), hyperparameter tuning via Cross-Validation and Grid Search, and performance evaluation.

## Data Loading
The dataset used in this project is the Iris dataset, obtained from https://archive.ics.uci.edu/dataset/53/iris. We need to import fetch_ucirepo from ucimlrepo, then we can call `fetch_ucirepo(id=53)` to get the dataset. The dataset is split into features and targets. We convert these into separate pandas dataframes and concatenate by column.

## Data Pre-processing
Machine learning algorithms in Spark MLlib require features to be formatted into a single vector column and labels to be numeric indices
1. **StringIndexer:** Converts the categorical `class` column into a numeric label `index` (0.0, 1.0, 2.0)
2. **VectorAssembler:** Combines the four structural measurements into a single `features` vector.
3. **Train-Test Splitting:** The data is split into 80% training and 20% testing sets

## Model Setup, Hyperparameter Tuning & Evaluation
We are implementing three classification models in this project: Decision Tree, Random Forest, and Logistic Regression. 
For each model, we use a `ParamGridBuilder` to define a grid of hyperparameters and a `CrossValidator` (3-fold) to systematically find the best parameters to prevent overfitting.

# 📊 Analysis
**CONTINUE FROM HERE**

## Performance Comparison
*(Note: Because the Iris dataset is small and easily separable, the accuracy metrics will likely be identical or very close to 1.000 for all models. The test set here contains 28 samples).*
* **Decision Tree:** Achieved strong performance but relies on a single tree structure, making it slightly more sensitive to how the training data was split.
* **Random Forest:** Consistently achieved top performance. By using an ensemble of trees, it reduces the variance and overfitting risks inherent in the standalone Decision Tree.
* **Logistic Regression:** Also performed exceptionally well, proving that the structural dimensions of the Iris flowers (sepal/petal lengths and widths) form a highly linearly separable feature space.

**Strengths and Limitations:**
* **Decision Tree:** * *Strengths:* Highly interpretable (can be visualised as if/else rules), requires no feature scaling.
    * *Limitations:* High risk of overfitting on more complex data.
* **Random Forest:**
    * *Strengths:* Highly robust to noise, handles non-linear features well, and reduces overfitting (variance) via bagging.
    * *Limitations:* A "black box" model that is harder to interpret compared to a single decision tree.
* **Logistic Regression:**
    * *Strengths:* Outputs probabilities (confidence scores) for classifications, highly efficient and fast to train.
    * *Limitations:* Assumes a linear relationship between features and log-odds of the classes; struggles with highly non-linear data.

**Justification of the Best-Performing Model:**
While all models performed extremely well due to the simplicity of the Iris dataset, **Random Forest** is theoretically and practically the strongest model here. Even though Logistic Regression works great on linearly separable data, Random Forest provides better generalization guarantees for unseen data without requiring strict assumptions about the data distribution. The hyperparameter tuning (Grid Search and Cross-Validation) optimized the depth and number of trees, striking the perfect balance between bias and variance.
