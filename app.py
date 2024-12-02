from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

app = Flask(__name__)

# Fetch the list of S&P 500 companies from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_list = pd.read_html(url, header=0)[0]
sp500_list['Serial Number'] = range(1, len(sp500_list) + 1)
sp500_list = sp500_list[['Serial Number', 'Symbol', 'Security']]

# Define the stock prediction function
def predict_stock_price(stock_symbol):
    # Fetch historical data
    data = yf.download(stock_symbol, start='2010-01-01', end='2020-01-01')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    # Create the MLP model
    model = Sequential()
    model.add(Dense(128, input_dim=train_data.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(train_data, train_data[:, 3], epochs=50, batch_size=32, validation_data=(test_data, test_data[:, 3]))
    
    # Predict the 'Close' price for the next day
    last_data = scaled_data[-1].reshape(1, -1)
    predicted_scaled_close_price = model.predict(last_data)
    
    # Extract and normalize the 'Close' prices separately for prediction
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit_transform(close_prices)
    predicted_close_price = scaler_close.inverse_transform(predicted_scaled_close_price)
    
    return predicted_close_price[0][0]

@app.route('/')
def index():
    return render_template('index.html', tables=[sp500_list.to_html(classes='data', header="true", index=False)])

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['symbol']
    predicted_price = predict_stock_price(stock_symbol)
    return render_template('result.html', symbol=stock_symbol, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
