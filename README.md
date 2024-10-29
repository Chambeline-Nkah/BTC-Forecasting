## **An Introduction to Time Series Forecasting:**

Time series forecasting is the process of analysing time series data using statistics and modelling to make predictions and inform strategic decision-making. It involves building models through historical analysis and using them to make observations and drive future strategic decision-making. This forecasting method helps identify trends, manage risks, and inform strategic planning, ultimately enhancing decision-making processes. 

Time series analysis is a useful tool to examine the characteristics of Bitcoin prices and returns and extract important statistics to forecast future values of the series. By forecasting BTC prices, investors can better navigate market fluctuations, identify optimal entry and exit points, and develop data-driven investment strategies.

## **Preprocessing Method:**

- **Data loading:** The bitcoin [dataset](https://www.kaggle.com/datasets/nisargchodavadiya/bitcoin-time-series-with-different-time-intervals?select=BTC-USD-15-MIN.csv) used was loaded from a CSV file containing a 15 minute interval data.
- **Checking and handling missing data:** The dataset had no missing values.
- **Timestamp conversion:** The timestamp column was in the Unix time format which is used in computing but yet it was difficult to be read. Thus, it was converted to a more human-readable datetime format. The timestamp column was converted to a datetime type and set it as the index.
- **Feature Selection:** We focus on the 'Close' price as the target variable for prediction.
- **Normalisation:** We apply min-max scaling to normalize the price data to a range of [0, 1].

```python
df = df = pd.read_csv("BTC-USD-15-MIN.csv")

# Converting the 'Timestamp' from Unix time to a readable datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

# Removing timezone info
df['Datetime'] = df['Datetime'].dt.tz_convert(None)

# Set 'Datetime' as the index of the DataFrame
df.set_index('Datetime', inplace=True)

# select close price
close_prices = df['Close'].values.reshape(-1, 1)

# normalize the close price data
scaler = MinMaxScaler()
close_price_scaled = scaler.fit_transform(close_prices)
```


## **Setting Up tf.data.Dataset for Model Inputs:**
We used TensorFlow's ```tf.data.Dataset``` API to create an efficient input pipeline:

1. We created sequences of 96 time steps (or 24 hours of 1-minute data) to predict the next time step.
2. The data is split into training (80%) and testing (20%) sets.
3. We use batching and prefetching to optimize performance.

```python
# Creating the dataset fxn
def create_dataset(data, time_steps=96):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

time_steps = 96 # 24 hours of 1 minute data

# Create datasets
X, y = create_dataset(close_price_scaled, time_steps)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Using tf.data.Dataset with shuffling and better batching
batch_size = 32
buffer_size = 1000

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
```

Importance of the above techniques in time series data:
- Batching allows for processing multiple data points simultaneously, improving computational efficiency and reducing training time.
- Windowing allows the model to learn from historical patterns.
- Prefetching ensures data is ready for the next training step, reducing idle time.


## **Model Architecture:**

A stacked LSTM network was used, and this is because:

```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()
```

- LSTM networks are well-suited for sequence prediction tasks and can capture long-term dependencies in time series data.
- The two LSTM layers allow the model to learn more complex patterns.
- ReLU activation helps mitigate the vanishing gradient problem.
- Dropout layer help to mitigate overfitting by randomly setting 20% of layer outputs to zero during training.
- Adam optimizer adapts the learning rate during training for better convergence.


## Results and Evaluation:

After training the model, we achieved the following performance metrics:

- Train MSE: 0.0003

- Test MSE: 0.0002

- Mean Absolute Error (MAE): 285.36

- Root Mean Square Error (RMSE): 366.72

Here is a visualization of the model's predictions compared to the actual Bitcoin prices:

![Alternate Text](btc_price_prediction_1.png)


Insights from the BTC Price Prediction plot:

- The plot shows that the model captured the general trend of Bitcoin prices over the period, as observed from the predicted prices (orange line) fluctuating and relatively close to the actual price (blue line). This indicates that the **LSTM model has effectively learned** from the historical data.


Another LSTM model was created with the tanh activation function and the dropout was increased to 30% and the model performed relatively well as compared to model 1 which made use of the relu activation function.


Here is a visualization of the model 2 predictions compared to the actual Bitcoin prices:

![Alternate Text](btc_price_prediction_2.png)

- The plot shows that model 2 effectively captured the general trend of Bitcoin prices over the period


## Conclusion

This project has been both rewarding and challenging at thesame time. This project was able to offer insights about the challenges and potential of applying deep learning to financial time series data.

- **Challenges encountered:**
  - High volatility of Bitcoin prices makes accurate prediction difficult.
  - Complex models like LSTM require careful tuning.
  - Training deep learning models can be resource-intensive, limiting the ability to experiment with different architectures.

- **Potential & future work:**
  - Experimenting with different architectures like GRU.
  - Optimizing hyperparameters.