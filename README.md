# 🌼 Description
This project uses machine learning models obtained from Spark MLlib (via pyspark) to classify species of Iris flowers according to their physical traits (sepal length, sepal width, petal length, and petal width). This project is done through Google Colab, in a python notebook.

# 📝 Methodology
The workflow of this project includes data loading, pre-processing, model setup (Decision tree, Random Forest, Logistic Regression), hyperparameter tuning via Cross-Validation and Grid Search, and performance evaluation.

## Data Loading
The dataset used in this project is the Iris dataset, obtained from the UCI machine learning repository at https://archive.ics.uci.edu/dataset/53/iris. We need to import fetch_ucirepo from ucimlrepo, then we can call `fetch_ucirepo(id=53)` to get the dataset. The dataset is split into features and targets. We convert these into separate pandas dataframes and concatenate by column. The dataset contains 150 samples of three species of Iris flowers, namely Setosa, Vriginica, and Versicolor. The four feaatures of each data point are the length and width of sepals and petals, in centimeters.

## Data Pre-processing
Machine learning algorithms in Spark MLlib require features to be formatted into a single vector column and labels to be numeric indices
1. **StringIndexer:** Converts the categorical `class` column into a numeric label `index` (0.0, 1.0, 2.0)
2. **VectorAssembler:** Combines the four structural measurements into a single `features` vector.
3. **Train-Test Splitting:** The data is split into 80% training and 20% testing sets

## Model Setup, Hyperparameter Tuning & Evaluation
We are implementing three classification models in this project: Decision Tree, Random Forest, and Logistic Regression. 
For each model, we use a `ParamGridBuilder` to define a grid of hyperparameters and a `CrossValidator` (3-fold) to systematically find the best parameters to prevent overfitting.

# 📊 Results
## Performance Comparison
*(Note: Because the Iris dataset is small and easily separable, the accuracy metrics will likely be identical or very close to 1.000 for all models. The test set here contains 28 samples).*


|           | Decision tree | Random Forest | Logistic Regression |
|-----------|---------------|---------------|---------------------|
| Accuracy  | 1.00          | 1.00          | 1.00                |
| F1-Score  | 1.00          | 1.00          | 1.00                |
| Precision | 1.00          | 1.00          | 1.00                |
| Recall    | 1.00          | 1.00          | 1.00                |

## Confusion Matrices
### Decision Tree
<img width="468" height="390" alt="image" src="https://github.com/user-attachments/assets/586ee909-10d9-46e1-b728-21110a24e8ef" />

### Random Forest
<img width="468" height="390" alt="image" src="https://github.com/user-attachments/assets/9b8b831d-892a-4f0b-87e3-8f0aa4f3f694" />

### Logistic Regression
<img width="468" height="390" alt="image" src="https://github.com/user-attachments/assets/d9f1765c-8b2d-44c3-9221-e381249c805f" />

## Strengths and Limitations
* **Decision Tree:**
   * *Strengths:* Easy to interpret as it imitates the logic of human thinking when making decisions (Dani and Ginting 2024)
   * *Limitations:* High risk of overfitting on more complex data.
* **Random Forest:**
    * *Strengths:* Highly robust to noise and reduces overfitting (Wu et al 2019)
    * *Limitations:* Hard to interpret.
* **Logistic Regression:**
    * *Strengths:* Outputs probabilities (confidence scores) for classifications, highly efficient and fast to train.
    * *Limitations:* Struggles with non-linear data because it assumes the data has a linear relationship. 

## Justification of the Best-Performing Model
While all models performed extremely well due to the simplicity of the Iris dataset, **Random Forest** is theoretically and practically the strongest model here. Even though Logistic Regression works great on linearly separable data, Random Forest works better on unknown data distributions without requiring the assumption that the data is linear.

# 👨‍🏫 Instructions to reproduce the analysis
*To replicate this PySpark machine learning workflow and achieve identical classification results, choose one of the execution methods below. Because local PySpark configurations on Windows or virtualized Hadoop environments frequently encounter network timeout limits, running this via **Google Colab** is highly recommended.*

## Option 1: Run in Google Colab (Recommended)
1. Go to [Google Colab](https://colab.research.google.com/).
2. Select the **Upload** tab from the splash screen (or navigate to `File > Upload notebook`).
3. Upload the **`P167590_STQD6324_IrisClassification.ipynb`** file included in this repository.
4. Click on the first code cell containing `!pip install pyspark` and run it to prepare the container environment.
5. Click on the cell containing `pip install ucimlrepo` and run it to get the method to import the dataset.
6. In the top navigation bar, select **Run all** to execute the entire code pipeline automatically.

---

## Option 2: Run in a Local Jupyter Environment (Desktop Method)
If you prefer running this script natively on your machine, complete the system prerequisites first to ensure the Spark engine handles dependencies correctly.

### 1. System Prerequisites
* **Python:** Python 3.8 or higher.
* **Java:** Java Development Kit (JDK) 8, 11, or 17 must be installed. *(PySpark officially does not support Java 21+ yet).* Ensure your system's `JAVA_HOME` environment variable is explicitly mapped.
* **Libraries:** Install pandas and pyspark via your terminal or command prompt:
  ```bash
  pip install pyspark pandas

### 2. Local Execution Steps
* Download the **`P167590_STQD6324_IrisClassification.ipynb`** file from this repository.
* Open Jupyter Notebook and navigate to where the .ipynb file is located.
* Click **Run > Run All Cells**
