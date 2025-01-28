import joblib
from sklearn.impute import KNNImputer
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import NeuralNetwork
from src.custom_dataset import CustomDataset
from src.training import train_model
from src.evaluation import evaluate_model
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import RobustScaler
import shap


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
df = load_data("./data/raw/houses.csv", "./data/raw/apartments.csv")
df = preprocess_data(df)

# Define target and split data
X, y = split_data(df, target="price")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

imputer = KNNImputer(n_neighbors=5)
impute_cols = ["state_of_building", "surface_of_the_plot", "nb_bedrooms", "living_area"]
imputer.fit(X_train[impute_cols])

joblib.dump(imputer, "./artifacts/encoders/knn_imputer.joblib")


new_num = ["surface_of_the_plot", "living_area", "nb_bedrooms"]

scaler = RobustScaler().fit(X_train[new_num])
joblib.dump(scaler, "./artifacts/encoders/numerical_scaler.joblib")
X_train[new_num] = scaler.transform(X_train[new_num])
X_test[new_num] = scaler.transform(X_test[new_num])


X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
cat_cols = [
    "district_id",
    "property_sub_type_id",
    "state_of_building",
    "equipped_kitchen",
    "garden",
    "swimming_pool",
    "terrace",
    "furnished",
]
numeric_cols = ["surface_of_the_plot", "living_area", "nb_bedrooms"]


# Create datasets
train_dataset = CustomDataset(X_train, y_train, numeric_cols, cat_cols)
test_dataset = CustomDataset(X_test, y_test, numeric_cols, cat_cols)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Model Initialization
model = NeuralNetwork(
    numeric_input_dim=len(numeric_cols),
    cat_input_dim=len(cat_cols),
    num_districts=len(df["district_id"].unique()),
    district_emb_dim=8,
    num_properties=len(df["property_sub_type_id"].unique()),
    property_emb_dim=4,
).to(device)


# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.001)
criterion = torch.nn.SmoothL1Loss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)
checkpoint_path = "./artifacts/checkpoints/best_model_checkpoint.pth"

# Train Model
train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    checkpoint_path,
    num_epochs=500,
    device=device,
)
# # Evaluate Model
loss, mae, r2, mape, smape, rmse = evaluate_model(
    model, test_loader, criterion, device, checkpoint_path
)

print(
    f"Final Test -> Loss: {loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, sMAPE: {smape:.4f}"
)

district_encoder_mapping = joblib.load("./artifacts/encoders/district_encoder.joblib")
property_sub_type_encode_mapping = joblib.load(
    "./artifacts/encoders/property_sub_type.joblib"
)


# SHAP Explanation
# def model_predict(data):
#     # data is a numpy array of shape (rows, numeric_cols + 2) because of district_id and property_sub_type_id
#     data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
#     num_numeric = len(numeric_cols)
#     num_categorical = len(cat_cols)

#     numeric_data = data_tensor[:, :num_numeric]
#     categorical_data = data_tensor[:, num_numeric : num_numeric + num_categorical]

#     dist_data = data_tensor[:, num_numeric].long()
#     prop_data = data_tensor[:, num_numeric + 1].long()

#     with torch.no_grad():
#         predictions = model(numeric_data, categorical_data, dist_data, prop_data)
#     return predictions.cpu().numpy()


# shap_explainer = shap.Explainer(
#     model_predict, X_train.values, feature_names=X_train.columns
# )
# shap_values = shap_explainer(X_test.values)
# shap.summary_plot(shap_values, X_test)
