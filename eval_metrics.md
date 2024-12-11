## Model declaration and instatiation

```python
class NeuralNetwork(nn.Module):
    def __init__(self, numeric_input_dim: int, num_districts: int, district_emb_dim: int,
                 num_properties: int, property_emb_dim: int, out_features: int = 1, dropout_rate: float = 0.05):
        """
        Initializes the neural network model, including embedding layers for district and property type.
        """
        super().__init__()
        self.district_embedding = nn.Embedding(num_districts, district_emb_dim)
        self.property_embedding = nn.Embedding(num_properties, property_emb_dim)

        total_input_dim = numeric_input_dim + district_emb_dim + property_emb_dim
        self.model = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
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
            nn.Linear(32, out_features)
        )


        model = NeuralNetwork(numeric_input_dim=len(numeric_cols), num_districts=len(df['district_id'].unique()),
                      district_emb_dim=16, num_properties=len(df['property_sub_type_id'].unique()),
                      property_emb_dim=8).to(device)

        # Optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
        criterion = torch.nn.SmoothL1Loss()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        checkpoint_path = "best_model_checkpoint.pth"
```

## Evaluation metrics:

Training-> Loss (SmoothL1): 58198.8662, MAE: 58097.3438, RMSE: 80472.1905, MAPE: 0.1954 ,sMAPE: 0.0382 ,R2: 0.6877

Test -> Loss(SmoothL1): 57364.9502, MAE: 57407.3203, R²: 0.6856, RMSE: 79122.5739, MAPE: 0.1963, sMAPE: 0.0385

## List of used features:

Initial features: living_area, state_of_building (Ordinal encoded), surface_of_the_plot, equipped_kitchen, nb_facades, nb_bedrooms

Added with feature engineering: price_m2_province

Added with Embeddings: district_id, property_sub_type_id

## About the final dataset:

- It has been scraped again to get more data
- Scraping has been done separately on apartments and houses and then two files have been merged together
- Data has been cleaned by removing non relevant columns (based on domain knowledge and retrospection after checking their influence with the shap explainer) and removing missing values from relevant columns. After a general 5% threshold has been applied, further missing values have been removed. This has been compensated with scraping more data.
  -Embeddings have been created for subtype of property and district category. The rationale is that it made much more sense to compact the information, that otherwise would have been one-hot-encoded (generating many more binary features uselessly incrementing model complexity), into a single feature (with the great value of each vector representing a category value having its own learnable parameters).
  -A different approach was used for state of the building feature, which has been ordinally encode as it implies an hierarchical value for its values.
