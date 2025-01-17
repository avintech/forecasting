import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class PricePredictionModel:
    def __init__(self, epochs=100, batch_size=32):
        self.label_encoders = {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.embedding_dims = {
            'index_main_type': 3,
            'index_freight_type': 3,
            'index_service_type': 3,
            'index_region': 3,
            'item_code': 5,
            'month': 3
        }

    def build_model(self, input_dims):
        """
        Build neural network architecture with embeddings for categorical features
        """
        # Input layers for each categorical feature
        inputs = {}
        embeddings = []
        
        # Create embedding layers for each categorical feature
        for feature, vocab_size in input_dims.items():
            inp = Input(shape=(1,), name=f'input_{feature}')
            inputs[feature] = inp
            
            # Create embedding layer
            emb = Embedding(vocab_size + 1,  # +1 for padding
                          self.embedding_dims[feature],
                          name=f'embedding_{feature}')(inp)
            emb = Flatten(name=f'flatten_{feature}')(emb)
            embeddings.append(emb)
        
        # Concatenate all embeddings
        concat = Concatenate()(embeddings)
        
        # Dense layers
        x = Dense(64, activation='relu')(concat)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        output = Dense(1, name='price')(x)
        
        # Create model
        model = Model(inputs=list(inputs.values()), outputs=output)
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def preprocess_data(self, df):
        """
        Preprocess the data by encoding categorical variables
        """
        df_processed = df.copy()
        
        # Define categorical columns
        categorical_columns = ['index_main_type', 'index_freight_type', 
                             'index_service_type', 'index_region', 'item_code',
                             'month']
        
        # Dictionary to store vocabulary sizes
        self.vocab_sizes = {}
        
        # Encode all categorical features
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                self.vocab_sizes[col] = len(self.label_encoders[col].classes_)
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def add_historical_features(self, df):
        """
        Add historical price features for each item
        """
        df = df.sort_values(['item_code', 'month'])
        
        # Add previous prices as features
        for i in range(1, 4):
            df[f'price_lag_{i}'] = df.groupby('item_code')['price'].shift(i)
        
        # Add rolling statistics
        df['rolling_mean_3m'] = df.groupby('item_code')['price'].rolling(window=3).mean().reset_index(0, drop=True)
        df['rolling_std_3m'] = df.groupby('item_code')['price'].rolling(window=3).std().reset_index(0, drop=True)
        
        return df
        
    def prepare_features(self, df_processed):
        """
        Prepare features for the model
        """
        df_with_history = self.add_historical_features(df_processed)
        df_complete = df_with_history.dropna()
        
        # Prepare inputs for each categorical feature
        # Add 'input_' prefix to match Input layer names
        inputs = {
            'input_' + feature: df_complete[feature].values
            for feature in ['index_main_type', 'index_freight_type', 
                          'index_service_type', 'index_region', 
                          'item_code', 'month']
        }
        
        return inputs, df_complete['price'].values
    
    def fit(self, df):
        """
        Fit the model on the training data
        """
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df_processed)
        
        # Build model using vocabulary sizes
        self.model = self.build_model(self.vocab_sizes)
        
        # Split data while maintaining input_ prefix
        X_train = {}
        X_val = {}
        y_train = None
        y_val = None
        
        # Get any key to use for splitting
        first_key = list(X.keys())[0]
        indices = np.arange(len(X[first_key]))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Split each feature
        for feature in X.keys():
            X_train[feature] = X[feature][train_idx]
            X_val[feature] = X[feature][val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
    
    def predict(self, df_new):
        """
        Make predictions for new data
        """
        # Preprocess new data
        df_processed = self.preprocess_data(df_new)
        
        # Prepare features
        X_new, _ = self.prepare_features(df_processed)
        
        # Make predictions
        predictions = self.model.predict(X_new)
        
        return predictions.flatten()

    
    def plot_training_history(self):
        """
        Plot training history
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training & validation MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Sample data creation (replace this with your actual data)
data = pd.read_csv('output.csv')

# Initialize and train model
model = PricePredictionModel(epochs=50, batch_size=16)
model.fit(data)

# Plot training history
model.plot_training_history()

# Make predictions
predictions = model.predict(data)
print("Sample predictions:", predictions)