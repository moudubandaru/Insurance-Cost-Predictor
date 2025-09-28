import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers, callbacks
from keras.layers import Dense
import joblib



health = pd.read_csv('insurance.csv')

health['sex'] = health['sex'].map({'male': 1, 'female': 0})
health['smoker'] = health['smoker'].map({'yes': 1, 'no': 0})
health['region'] = health['region'].map({'southwest': 0, 'southeast': 1,'northeast': 2, 'northwest': 3})


data_train = health.sample(frac=0.75, random_state=0)
data_valid = health.drop(data_train.index)



X_train = data_train.drop('charges', axis=1)
X_valid = data_valid.drop('charges', axis=1)


y_train = data_train['charges'] 
y_valid = data_valid['charges']

scaler = MinMaxScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)


#------Data Assignment & pre-processing---------

early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 5,
    restore_best_weights = True
) 


model = keras.Sequential([
    layers.Dense(256, activation = 'relu', input_shape = [6]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer = 'adam',
    loss = 'mae'
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data = (X_valid_scaled, y_valid),
    batch_size = 256,
    epochs = 1000,
    callbacks = [early_stopping]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
plt.show()
model.save("health_model.keras")
joblib.dump(scaler, 'scaler.pkl')