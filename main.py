import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_preprocessing import load_data, preprocess_data, split_data
from model import NeuralNetwork
from custom_dataset import CustomDataset
from training import train_model
from evaluation import evaluate_model
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import RobustScaler
import shap



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
df = load_data('./data/houses.csv', './data/apartments.csv')
df = preprocess_data(df)

# Define target and split data
X, y = split_data(df, target='price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
new_num = ['surface_of_the_plot','living_area','nb_facades','nb_bedrooms']
for col in new_num:
    scaler = RobustScaler().fit(X_train[[col]])
    X_train[col] = scaler.transform(X_train[[col]])
    X_test[col] = scaler.transform(X_test[[col]])



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

numeric_cols = [c for c in X_train.columns if c not in ['district_id', 'property_sub_type_id']]
cat_cols = ['district_id', 'property_sub_type_id']


# Create datasets
train_dataset = CustomDataset(X_train, y_train, numeric_cols, cat_cols)
test_dataset = CustomDataset(X_test, y_test, numeric_cols, cat_cols)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=85, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=85, shuffle=False)

# Model Initialization
model = NeuralNetwork(numeric_input_dim=len(numeric_cols), num_districts=len(df['district_id'].unique()), 
                      district_emb_dim=16, num_properties=len(df['property_sub_type_id'].unique()), 
                      property_emb_dim=8).to(device)

# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.00001)
criterion = torch.nn.SmoothL1Loss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
checkpoint_path = "best_model_checkpoint.pth"

# Train Model
train_model(model, train_loader, criterion, optimizer,scheduler, checkpoint_path, num_epochs=300, device=device)
# Evaluate Model
loss, mae, r2,mape,smape,rmse = evaluate_model(model, test_loader, criterion, device,checkpoint_path)

print(f"Final Test -> Loss: {loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, sMAPE: {smape:.4f}"    )

# SHAP Explanation
def model_predict(data):
    # data is a numpy array of shape (rows, numeric_cols + 2) because of district_id and property_sub_type_id
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    numeric_data = data_tensor[:, :len(numeric_cols)]
    dist_data = data_tensor[:, len(numeric_cols)].long()
    prop_data = data_tensor[:, len(numeric_cols)+1].long()

    with torch.no_grad():
        predictions = model(numeric_data, dist_data, prop_data)
    return predictions.cpu().numpy()

shap_explainer = shap.Explainer(model_predict, X_train.values, feature_names=X_train.columns)
shap_values = shap_explainer(X_test.values)
shap.summary_plot(shap_values, X_test)
