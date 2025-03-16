## Forecasting High Incomes with US Adult Census Insights

## Project Overview
This project aims to develop a predictive model that determines whether an individual's yearly income exceeds $50,000 based on a dataset obtained from the US adult census. The model leverages various socioeconomic variables such as age, education, employment, and marital status. The study provides valuable insights for economic studies, targeted marketing, and public policy formulation.

## Dataset
The dataset used for this project was obtained from Kaggle and consists of 32,561 unique records. The target variable, `income`, classifies individuals into two income categories: `>=50K` and `<50K`.

Data Cleaning & Preprocessing
- Handling Missing Values: Special characters representing missing values were replaced with NaN and imputed using the mode for categorical features.
- Duplicate Removal: Duplicate entries were detected and removed to maintain data integrity.
- Outlier Handling: Outliers in numerical columns were identified and treated using the Interquartile Range (IQR) and Z-score methods.
- Feature Engineering: Created a new feature, `income_per_hour`, to measure the relationship between capital gains and work hours.
- Encoding Categorical Variables: Used mapping and one-hot encoding to convert categorical features into numerical values.
- Feature Selection: Removed low-correlation features to optimize model efficiency.
- Normalization: Applied Min-Max scaling to ensure uniformity across features.

## Exploratory Data Analysis (EDA)
- Visualized data distributions using histograms, box plots, and violin plots.
- Conducted correlation analysis to identify relationships between variables.
- Used KDE plots and pairplots to explore feature interactions.

## Machine Learning Models
We trained and evaluated seven different classification models:
1. Logistic Regression
2. K-Nearest Neighbors (K-NN)
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. Decision Tree Classifier
6. Naive Bayes
7. XGBoost Classifier

### Model Performance
The models were evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. The best-performing model was XGBoost, achieving an accuracy of 86.7% after hyperparameter tuning.

## Results
- The most significant features affecting income prediction were `education`, `capital gain`, and `hours per week`.
- Random Forest and XGBoost provided the most reliable predictions with high accuracy and interpretability.
- Feature importance analysis helped identify key socioeconomic determinants of income disparities.

## Installation & Usage
### Prerequisites
- Python 3.x
- Jupyter Notebook / Google Colab
- Required Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/adult-income-prediction.git
   cd adult-income-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
4. Open `income_prediction.ipynb` and execute the cells step by step.

## Future Improvements
- Fine-tuning hyperparameters using GridSearchCV for all models.
- Implementing deep learning models for improved accuracy.
- Exploring additional socioeconomic datasets to enhance model generalization.

## Contributors
- Areef Shaik
- Jahnavi Pravaleeka Battu

## References
1. [US Adult Census Dataset](https://www.kaggle.com/datasets/jainaru/adult-income-census-dataset)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
3. [Scikit-Learn Documentation](https://scikit-learn.org/stable/)



