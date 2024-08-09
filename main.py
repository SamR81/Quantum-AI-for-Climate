import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load your dataset
data = pd.read_csv('dataset.csv')

# Preprocessing the Data
feature_columns = ['feature1', 'feature2', 'feature3']  # Example feature names
target_column = 'temperature_anomaly'  # Example target column name

# Normalize the features
scaler = MinMaxScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Splitting the data into training and testing sets
X = data[feature_columns].values
y = data[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for Conv1D input (timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the Model
model = tf.keras.Sequential()

# Conv1D layer (no TimeDistributed)
model.add(tf.keras.layers.Conv1D(64, 2, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
model.add(tf.keras.layers.Flatten())

# Reshape for LSTM
model.add(tf.keras.layers.Reshape((1, -1)))  # Reshape to (batch_size, timesteps=1, features)
model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=False))

# Output layer
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Plot Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Temperature Anomaly')
plt.title('Predicted vs Actual Temperature Anomalies')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Create an Animation of Temperature Anomalies Over Time
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, len(y_test))
ax.set_ylim(min(y_test), max(y_test))
ax.set_title('Temperature Anomalies Over Time')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = list(range(frame))
    y = y_test[:frame]
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(y_test), init_func=init, blit=True)
ani.save('temperature_anomalies.gif', writer='imagemagick')  # Save as a GIF
plt.show()
