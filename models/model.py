import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Dense, Dropout, Attention, Embedding, Flatten, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import time

# Load Pickle Dataset Correctly
with open(r'C:\Users\dhawa\OneDrive\Desktop\dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Ensure dataset is a DataFrame
if isinstance(data, dict):
    data = pd.DataFrame(data)

# Identify label column and encode it
label_column = 'type'  # Adjust based on dataset structure
label_encoder = LabelEncoder()
data[label_column] = label_encoder.fit_transform(data[label_column])

# Encode 'configuration' column for environment-aware learning
config_encoder = LabelEncoder()
data['configuration'] = config_encoder.fit_transform(data['configuration'])

# Drop unnecessary columns
features = data.drop(columns=['day', 'object_id', 'position'])
labels = data[label_column].values
configs = data['configuration'].values

# Normalize Features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert data into sequences for movement tracking
sequence_length = 10  # How many time steps to track
X_sequences, y_sequences, config_sequences = [], [], []
for i in range(len(features) - sequence_length):
    X_sequences.append(features[i:i + sequence_length])
    y_sequences.append(labels[i + sequence_length])
    config_sequences.append(configs[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
config_sequences = np.array(config_sequences)

# Split Dataset
X_train, X_test, y_train, y_test, config_train, config_test = train_test_split(X_sequences, y_sequences, config_sequences, test_size=0.2, random_state=42)

# Define Model with Time-Series Movement Tracking
input_layer = Input(shape=(sequence_length, X_train.shape[2]))
x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(32))(x)

# Attention Mechanism
attention = Attention()([x, x])
x = Dense(32, activation='relu')(attention)
x = Dropout(0.3)(x)

# Environment-Aware Learning (Configuration Embedding)
config_input = Input(shape=(1,))
config_embedding = Embedding(input_dim=len(np.unique(configs)), output_dim=5)(config_input)
config_embedding = Flatten()(config_embedding)

# Merge with Main Network
merged = tf.keras.layers.Concatenate()([x, config_embedding])

# Output Layer (Classification)
classification_output = Dense(len(np.unique(labels)), activation='softmax', name='classification_output')(merged)

# Define Model
model = Model(inputs=[input_layer, config_input], outputs=classification_output)

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit([X_train, config_train], y_train,
          epochs=20, batch_size=32, validation_data=([X_test, config_test], y_test))

# Save Model
model.save("wifi_csi_security_model.h5")

print("Model training complete and saved as wifi_csi_security_model.h5")

# Evaluation
loss, accuracy = model.evaluate([X_test, config_test], y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Real-Time Object Presence & Movement Detection
def detect_object_movement(new_data, config_data, threshold=0.005):
    predictions = model.predict([new_data, config_data])
    max_probs = np.max(predictions, axis=1)
    anomalies = np.where(max_probs < threshold)[0]
    
    if len(anomalies) > 0:
        print("⚠️ ALERT: Unexpected Object Presence or Movement Detected!")
    else:
        print("✅ Normal Object Detection.")

# Example Real-Time Detection Loop
print("Listening for real-time CSI object movement detection...")
while True:
    # Simulating real-time CSI data input (replace with real CSI streaming)
    simulated_data = np.random.normal(size=(1, sequence_length, X_train.shape[2]))
    simulated_config = np.array([[0]])  # Assuming configuration 0 for now
    
    detect_object_movement(simulated_data, simulated_config)
    time.sleep(2)  # Simulating a delay between CSI data frames
