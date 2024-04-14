from typing import Optional

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

input_data = np.random.randint(0, 100, size=(10000, 3))
labels = np.sum(input_data, axis=1)

def fit_model(model: Sequential, input_data: np.array, output: np.array, optimizer: str = 'adam',
              loss: str = 'mean_squared_error', epochs: int = 10, save_model: Optional[str] = None):
    model.compile(optimizer, loss)
    model.fit(input_data, output, epochs=epochs, batch_size=32)

    if save_model:
        model.save(f'{save_model}.keras')
    return model


def read_model(filepath: str) -> Sequential:
    return load_model(filepath)


model = Sequential([
    Dense(30, input_dim=3, activation='relu'),
    Dense(1)
])
fit_model(model, input_data, labels, epochs=15,
          # save_model='sum0_100'
          )

# model = read_model(f'sum.keras')



predict = model.predict([[10, 20, 99]])

print(predict)