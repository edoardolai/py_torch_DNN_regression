# pytorch_DNN_regression

This project is a regression model aimed at predicting apartment and house prices based on various features. The model uses a deep learning approach, specifically a neural network, to make predictions. The project processes a dataset consisting of real estate listings, performs data preprocessing, and trains a neural network model to predict the price of properties.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- pip
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- torchmetrics

### Install Dependencies

To install the required Python packages, run the following command:

```bash
pip install -r requirements.txt
```
## Data

The project uses two datasets (houses.csv and apartments.csv), which are assumed to contain real estate data such as price, living_area, surface_of_the_plot, district, property_sub_type, etc. The datasets are concatenated to form a unified DataFrame for training and testing.

Make sure the houses.csv and apartments.csv files are available in the data/ folder.

Running the Project

### Step 1: Data Preprocessing

The data_preprocessing.py file handles the data loading, cleaning, and feature engineering. It performs the following operations:
	•	Merges houses.csv and apartments.csv.
	•	Removes columns that are irrelevant to the model.
	•	Fills missing values and applies transformations.
	•	One-hot encodes categorical features.
	•	Scales numeric features.
	•	Creates the necessary features for training (like avg_monthly_income_per_district and avg_monthly_income_province).

### Step 2: Model Training

To train the model, you can run the main.py script. It does the following:
	1.	Loads and preprocesses the data.
	2.	Splits the data into training and test sets.
	3.	Defines the neural network architecture using the NeuralNetwork class in model.py.
	4.	Trains the model using the data.
	5.	Saves the best model based on validation loss.
	6.	Evaluates the model on the test set.

Run the following command to start training:
```bash
python main.py
```

### Step 3: Model Evaluation

Once training is complete, the model will be evaluated on the test set. The evaluation includes computing metrics such as:
	•	Loss (L1Loss)
	•	Mean Absolute Error (MAE)
	•	R² Score

These metrics help assess the performance of the trained model.

### Hyperparameter Tuning

You can experiment with the following hyperparameters to improve model performance:
	•	Learning rate (lr)
	•	Dropout rate
	•	Batch size
	•	Number of epochs
	•	Number of layers and units in the neural network

Adjust these parameters in main.py and experiment with different configurations to achieve better performance.

### Acknowledgements
	•	This project uses PyTorch for neural network modeling.
	•	SHAP is used for model interpretability.
	•	Scikit-learn is used for data preprocessing and splitting.

### Additional Notes:
**Requirements**: Make sure that you have the required dependencies in your `requirements.txt`. If you don’t have it, you can create one by running:
   ```bash
   pip freeze > requirements.txt
   ```

