import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score, MeanAbsoluteError
import torch.optim.lr_scheduler as lr_scheduler
import shap
import os

# Load data
houses = pd.read_csv('houses.csv')
apartments = pd.read_csv('apartments.csv')

df = pd.concat([houses, apartments], axis=0)

# Initial cleaning
df.drop(['Unnamed: 0','terrace_surface','garden_surface','id','locality'], axis=1, inplace=True)

# Drop rows with too many missing values
threshold = 0.05 * len(df)
columns_low_missing_data = df.columns[df.isna().sum() < threshold]
df.dropna(subset=columns_low_missing_data, inplace=True)

# Drop duplicates
df.drop_duplicates(subset=['zip_code', 'latitude', 'longitude'], keep='first', inplace=True)

# One-hot encode property_type (This differentiates house vs apartment)
df = pd.concat([df, pd.get_dummies(df["property_type"])], axis=1)
df.drop('property_type', axis=1,inplace=True)

# Province classification based on zip_code patterns
provinces = [
    'Bruxelles_province','Brabant_Wallon_province', 'Brabant_Flamand_province', 
    'Anvers_province', 'Limbourg_province','Liège_province','Namur_province',
    'Hainaut_province','Luxembourg_province','Flandre_Occidentale_province','Flandre_Orientale_province'
]

brussels_capital = '^10|^11|^12'
walloon_brabant = '^13|^14'
flemish_brabant = '^15|^16|^17|^18|^19|^30|^31|^32|^33|^34'
antwerp = '^20|^21|^22|^23|^24|^25|^26|^27|^28|^29'
limburg = '^35|^36|^37|^38|^39'
liège = '^40|^41|^42|^43|^44|^45|^46|^47|^48|^49'
namur = '^50|^51|^55|^53|^55|^55|^56|^57|^58|^59'
hainaut = '^60|^61|^62|63|^64|^65|^70|^71|^77|^73|^74|^75|^76|^77|^78|^79'
luxembourg = '^66|^67|^68|^69'
west_flanders = '^80|^81|^82|^83|^84|^85|^86|^87|^88|^89'
east_flanders = '^90|^91|^92|^93|^94|^95|^96|^97|^98|^99'

conditions = [
    df["zip_code"].astype(str).str.contains(brussels_capital),
    df["zip_code"].astype(str).str.contains(walloon_brabant),
    df["zip_code"].astype(str).str.contains(flemish_brabant),
    df["zip_code"].astype(str).str.contains(antwerp),
    df["zip_code"].astype(str).str.contains(limburg),
    df["zip_code"].astype(str).str.contains(liège),
    df["zip_code"].astype(str).str.contains(namur),
    df["zip_code"].astype(str).str.contains(hainaut),
    df["zip_code"].astype(str).str.contains(luxembourg),
    df["zip_code"].astype(str).str.contains(west_flanders),
    df["zip_code"].astype(str).str.contains(east_flanders),
]

df["province"] = np.select(conditions, provinces, default='Other')

avg_monthly_income_province = {
    'Bruxelles_province': 4.748,
    'Anvers_province': 4.160,
    'Brabant_Flamand_province': 4.367,
    'Brabant_Wallon_province': 4.272,
    'Flandre_Occidentale_province': 3.684,
    'Flandre_Orientale_province': 3.903,
    'Hainaut_province': 3.694,
    'Liège_province': 3.792,
    'Limbourg_province': 3.778,
    'Luxembourg_province': 3.457,
    'Namur_province': 3.670,
}
df['avg_monthly_income_province'] = df['province'].map(avg_monthly_income_province)

avg_monthly_income_dict = {
    'Anvers': 4.213, 'Malines': 4.121, 'Turnhout': 4.031, 'Bruxelles': 4.748,
    'Hal-Vilvorde': 4.406, 'Louvain': 4.347, 'Nivelles': 4.272, 'Bruges': 3.802,
    'Dixmude': 3.510, 'Ypres': 3.564, 'Courtrai': 3.780, 'Ostende': 3.771,
    'Roulers': 3.626, 'Tielt': 3.626, 'Furnes': 3.264, 'Alost': 3.637,
    'Termonde': 3.714, 'Eeklo': 3.502, 'Gand': 4.076, 'Audenarde': 3.562,
    'Saint-Nicolas': 3.874, 'Ath': 3.382, 'Charleroi': 3.893, 'Mons': 3.764,
    'Soignies': 3.678, 'Thuin': 3.294, 'Tournai': 3.404, 'Mouscron': 3.404,
    'Huy': 4.073, 'Liège': 3.836, 'Verviers': 3.572, 'Waremme': 3.580,
    'Hasselt': 3.913, 'Maaseik': 3.561, 'Tongres': 3.540, 'Arlon': 3.459,
    'Bastogne': 3.221, 'Marche-en-Famenne': 3.217, 'Neufchâteau': 3.385,
    'Virton': 3.683, 'Dinant': 3.134, 'Namur': 3.743, 'Philippeville': 3.309
}
df['avg_monthly_income_per_district'] = df['district'].map(avg_monthly_income_dict)
prix_m2_app={
    'Flandre_Orientale_province': 2864,
    'Anvers_province': 2789,
    'Bruxelles_province': 3401 ,
    'Liège_province': 2214,
    'Brabant_Flamand_province': 3197,
    'Hainaut_province': 1854,
    'Brabant_Wallon_province': 3156,
    'Luxembourg_province': 2395,
    'Limbourg_province': 2488 ,
    'Namur_province': 2543,
    'Flandre_Occidentale_province': 3759
}
prix_m2_house={
    'Flandre_Orientale_province': 2229,
    'Anvers_province': 2365,
    'Bruxelles_province': 3245 ,
    'Liège_province': 1684,
    'Brabant_Flamand_province': 2484,
    'Hainaut_province': 1382,
    'Brabant_Wallon_province': 2302,
    'Luxembourg_province': 1574,
    'Limbourg_province': 1898 ,
    'Namur_province': 1625,
    'Flandre_Occidentale_province': 2017
}

df['price_m2_province'] = df.apply(
    lambda x: prix_m2_house[x['province']] if x['HOUSE'] == 1 else prix_m2_app[x['province']],
    axis=1
)
# For apartments, fill surface_of_the_plot with 0
df.loc[df["APARTMENT"] == 1, "surface_of_the_plot"] = df.loc[df["APARTMENT"] == 1, "surface_of_the_plot"].fillna(0)

# Remove rows that still have null in key columns
df.dropna(subset=['avg_monthly_income_per_district'], inplace=True)

# Remove outliers for price on the combined dataset
def remove_outliers(data, columns):
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

df = remove_outliers(df.copy(), ['price','living_area','surface_of_the_plot'])

# Encode state_of_building
states = ['To renovate','To be done up','To restore','Good','Just renovated','As new']
ordinal_encoder = OrdinalEncoder(categories=[states])
state = df['state_of_building']
state_not_null = state[state.notnull()].values.reshape(-1,1)
state_encoded = ordinal_encoder.fit_transform(state_not_null)
df.loc[state.notnull(), 'state_of_building'] = np.squeeze(state_encoded)



# Drop rows with missing essential features
df.dropna(subset=['latitude','longitude','state_of_building','nb_facades','living_area','surface_of_the_plot'], inplace=True)

# Let's encode district and property_sub_type using LabelEncoder
district_encoder = LabelEncoder()
property_sub_type_encoder = LabelEncoder()

df['district_id'] = district_encoder.fit_transform(df['district'].astype(str))
df['property_sub_type_id'] = property_sub_type_encoder.fit_transform(df['property_sub_type'].astype(str))

# Now drop the original string columns (if desired)
df.drop(['district', 'property_sub_type'], axis=1, inplace=True)

# Define target
target = 'price'
X = df.drop(target, axis=1)
y = df[target]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Numeric columns to scale (excluding the categorical IDs)
new_num = ['surface_of_the_plot','living_area','nb_facades','nb_bedrooms']

# drop_cols = ['HOUSE','APARTMENT','Apartment_Block', 'Bungalow', 'Castle',
#     'Chalet', 'Country_Cottage', 'Exceptional_Property', 'Farmhouse',
#     'House', 'Manor_House', 'Mansion', 'Mixed_Use_Building',
#     'Other_Property', 'Town_House', 'Villa','furnished','swimming_pool','fireplace','terrace','garden','equipped_kitchen']

# # Also drop district and province one-hot columns if needed (adjust as necessary)
# drop_cols += ['Alost', 'Anvers', 'Arlon', 'Ath', 'Audenarde', 'Bastogne', 'Bruges',
#        'Bruxelles', 'Charleroi', 'Courtrai', 'Dinant', 'Dixmude', 'Eeklo',
#        'Furnes', 'Gand', 'Hal-Vilvorde', 'Hasselt', 'Huy', 'Liège', 'Louvain',
#        'Maaseik', 'Malines', 'Mons', 'Mouscron', 'Namur', 'Neufchâteau',
#        'Nivelles', 'Ostende', 'Philippeville', 'Roulers', 'Saint-Nicolas',
#        'Soignies', 'Termonde', 'Thuin', 'Tielt', 'Tongres', 'Tournai',
#        'Turnhout', 'Verviers', 'Virton', 'Waremme', 'Ypres', 'Anvers_province',
#        'Brabant_Flamand_province', 'Brabant_Wallon_province',
#        'Bruxelles_province', 'Flandre_Occidentale_province',
#        'Flandre_Orientale_province', 'Hainaut_province', 'Limbourg_province',
#        'Liège_province', 'Luxembourg_province', 'Namur_province']

# X_train.drop(columns=[c for c in drop_cols if c in X_train.columns], axis=1, inplace=True)
# X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], axis=1, inplace=True)
(X_train.columns)
X_train.drop(['province','latitude','longitude','zip_code','APARTMENT','HOUSE', 'garden','furnished','swimming_pool','fireplace','avg_monthly_income_per_district','terrace','avg_monthly_income_province'], inplace=True,axis=1)
X_test.drop(['province','latitude','longitude','zip_code','APARTMENT','HOUSE', 'garden','furnished','swimming_pool','fireplace','avg_monthly_income_per_district','terrace','avg_monthly_income_province'], inplace=True,axis=1)

for col in new_num:
    scaler = RobustScaler().fit(X_train[[col]])
    X_train[col] = scaler.transform(X_train[[col]])
    X_test[col] = scaler.transform(X_test[[col]])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

numeric_cols = [c for c in X_train.columns if c not in ['district_id', 'property_sub_type_id']]
cat_cols = ['district_id', 'property_sub_type_id']

class CustomDataset(Dataset):
    def __init__(self, X, y, numeric_cols, cat_cols):
        self.X_numeric = torch.tensor(X[numeric_cols].values, dtype=torch.float32)
        self.X_district = torch.tensor(X['district_id'].values, dtype=torch.long)
        self.X_property = torch.tensor(X['property_sub_type_id'].values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_numeric[idx], self.X_district[idx], self.X_property[idx], self.y[idx]

training_dataset = CustomDataset(X_train, y_train, numeric_cols, cat_cols)
test_dataset = CustomDataset(X_test, y_test, numeric_cols, cat_cols)

training_loader = DataLoader(training_dataset, shuffle=True, batch_size=85) # Slightly larger batch size
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=85)

# Determine embedding sizes
num_districts = X['district_id'].nunique()
num_property_types = X['property_sub_type_id'].nunique()

district_emb_dim = 16
property_emb_dim = 8

class NeuralNetwork(nn.Module):
    def __init__(self, numeric_input_dim, num_districts, district_emb_dim, num_properties, property_emb_dim, out_features=1, dropout_rate=0.2):
        super().__init__()
        # Embeddings
        self.district_embedding = nn.Embedding(num_districts, district_emb_dim)
        self.property_embedding = nn.Embedding(num_properties, property_emb_dim)

        total_input_dim = numeric_input_dim + district_emb_dim + property_emb_dim

        # A simpler architecture with batch norm and dropout
        self.model = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, out_features)
        )

    def forward(self, numeric_x, district_ids, property_ids):
        dist_emb = self.district_embedding(district_ids)
        prop_emb = self.property_embedding(property_ids)
        combined = torch.cat([numeric_x, dist_emb, prop_emb], dim=1)
        return self.model(combined)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNetwork(
    numeric_input_dim=len(numeric_cols), 
    num_districts=num_districts, district_emb_dim=district_emb_dim,
    num_properties=num_property_types, property_emb_dim=property_emb_dim,
    out_features=1, 
    dropout_rate=0.1
).to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0001) # Increased weight_decay and lowered lr slightly
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
r2_score = R2Score().to(device)
mae = MeanAbsoluteError().to(device)

num_epochs = 300
best_train_loss = float('inf')
patience = 30
no_improve_count = 0
checkpoint_path = "best_model_checkpoint.pth"
print(X_train.shape[0])
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    r2_score.reset()
    mae.reset()
    for numeric_x, dist_ids, prop_ids, batch_y in training_loader:
        numeric_x, dist_ids, prop_ids, batch_y = numeric_x.to(device), dist_ids.to(device), prop_ids.to(device), batch_y.to(device).unsqueeze(1)

        train_outputs = model(numeric_x, dist_ids, prop_ids)
        # batch_y = torch.exp(batch_y) - 1
        loss = criterion(train_outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        r2_score.update(train_outputs, batch_y)
        mae.update(train_outputs, batch_y)
    
    avg_loss = epoch_loss / len(training_loader)
    avg_r2 = r2_score.compute().item()
    avg_mae = mae.compute().item()
    scheduler.step(avg_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}')

    # Early Stopping
    if avg_loss < best_train_loss:
        best_train_loss = avg_loss
        no_improve_count = 0
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_train_loss': best_train_loss,
            'mae': mae,
            'r2_score': avg_r2,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"New best model at epoch {epoch+1}, Loss: {best_train_loss:.4f}")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

# Once training is finished or early stopped, load the best model
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print("Best model loaded.")

# Final evaluation on the test set (if desired)
model.eval()
epoch_loss = 0
r2_score.reset()
mae.reset()
with torch.no_grad():
    for numeric_x, dist_ids, prop_ids, batch_y in test_loader:
        numeric_x, dist_ids, prop_ids, batch_y = numeric_x.to(device), dist_ids.to(device), prop_ids.to(device), batch_y.to(device).unsqueeze(1)
        test_outputs = model(numeric_x, dist_ids, prop_ids)
        loss = criterion(test_outputs, batch_y)
        epoch_loss += loss.item()
        r2_score.update(test_outputs, batch_y)
        mae.update(test_outputs, batch_y)

avg_loss = epoch_loss / len(test_loader)
avg_r2 = r2_score.compute().item()
avg_mae = mae.compute().item()
print(f'Final Test -> Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}')

# SHAP or further analysis as you wish
# SHAP Explanation
def model_predict(data):
    # data is a numpy array of shape (rows, numeric_cols + 2) because of district_id and property_sub_type_id
    # We must split it just like in forward
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    numeric_data = data_tensor[:, :len(numeric_cols)]
    dist_data = data_tensor[:, len(numeric_cols)].long()
    prop_data = data_tensor[:, len(numeric_cols)+1].long()

    with torch.no_grad():
        predictions = model(numeric_data, dist_data, prop_data)
    return predictions.cpu().numpy()

# Prepare data for SHAP (ensure district_id and property_sub_type_id are last two columns of X_train)
shap_explainer = shap.Explainer(model_predict, X_train.values, feature_names=X_train.columns)
shap_values = shap_explainer(X_test.values)
shap.summary_plot(shap_values, X_test)


# batch size 32
# Final Test -> Loss: 7033247581.2713, MAE: 62478.4023, R²: 0.6916 batch size 32

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32#
# Final Test -> Loss: 8025751845.7054, MAE: 68916.4219, R²: 0.6480 

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#  Final Test -> Loss: 7010295650.8070, MAE: 61492.5430, R²: 0.6771 

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#updated learning rate-> 0.004 from 0.002 
# Final Test -> Loss: 6792519073.6842, MAE: 61092.0469, R²: 0.6871

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#updated learning rate-> 0.005 from 0.004
#decreased dropout rate->0.1 from 0.2
# Final Test -> Loss: 6789266088.4211, MAE: 59943.8828, R²: 0.6873

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#updated learning rate-> 0.005 from 0.004
#decreased dropout rate->0.1 from 0.2
#removed one dropout layer
#1 Final Test -> Loss: 6789266088.4211, MAE: 59943.8828, R²: 0.6873
#2 Final Test -> Loss: 6384215980.9123, MAE: 58921.4375, R²: 0.7059

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#updated learning rate-> 0.005 from 0.004
#decreased dropout rate->0.1 from 0.2
#removed one dropout layer
#batchsize16 -> worse
# Final Test -> Loss: 7157548550.1754, MAE: 62190.1406, R²: 0.6701

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#updated learning rate-> 0.005 from 0.004
#decreased dropout rate->0.1 from 0.2
#removed one dropout layer
#batchsize64 -> better
# Final Test -> Loss: 6373032636.6316, MAE: 58472.4766, R²: 0.7068

#removed features  'garden','furnished','swimming_pool','fireplace',batch size 32, removing outliers from price, living area and surface of the plot
#updated learning rate-> 0.005 from 0.004
#decreased dropout rate->0.1 from 0.2
#removed one dropout layer
#batchsize 85
# Final Test -> Loss: 58519.7793, MAE: 58511.4688, R²: 0.7033