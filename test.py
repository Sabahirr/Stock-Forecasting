import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import matplotlib.pyplot as plt


icon = Image.open('icon.png')
logo = Image.open('logo.png')
banner = Image.open('banner.jpg') 

st.set_page_config(layout = 'wide' ,
                   page_title="Sabahir's app" ,
                   page_icon = icon)
st.title('Stock Price Forecasting')
st.text('Simple Machine Learning Web Application with Streamlit')

# Sidebar Container
st.sidebar.image(image = logo)
menu = st.sidebar.selectbox('', ['Homepage' ,'Model', 'Evaluation'])


if menu == 'Homepage':
    print('xosh geldin')

elif menu == 'Model':
    sub_menu = st.sidebar.selectbox('', ['Evaluation','Homepage' ,'Model' ])
    
    def get_stock_data(ticker, interval):
        stock_info = yf.Ticker(ticker)
        stock_history = stock_info.history(period="max")
        start_date = stock_history.index.min().strftime('%Y-%m-%d')
        end_date = stock_history.index.max().strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        return stock_data['Close']

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    def forecast_stock(model, data, scaler, time_step, n_periods):
        x_input = data[-time_step:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        n_steps = time_step
        i = 0
        while(i < n_periods):
            if(len(temp_input) > time_step):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        forecasted_stock_prices = scaler.inverse_transform(lst_output)
        return forecasted_stock_prices

    st.image(banner,use_column_width = 'always')
    entered = st.text_input('Enter the stock ticker or select the most popular one:')
    most_popular = st.selectbox('Most popular stock:', ['BTC-USD','S&P 500', 'Nasdaq', 'Bitcoin USD'])
    ticker = entered or most_popular 
    forecast_type = st.selectbox('Select forecast type:', ['Day', 'Week', 'Month'])

    if forecast_type == 'Day':
        interval = '1d'
        resample_interval = 'D'
        n_periods = 30
    elif forecast_type == 'Week':
        interval = '1wk'
        resample_interval = 'W'
        n_periods = 10
    elif forecast_type == 'Month':
        interval = '1mo'
        resample_interval = 'M'
        n_periods = 5
    else:
        st.error("Invalid forecast type. Please select 'Day', 'Week', or 'Month'.")

    if st.button('Fetch and Forecast'):
        data = get_stock_data(ticker, interval)
        data = data.resample(resample_interval).ffill()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        time_step = 100
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=1,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        forecasted_prices = forecast_stock(model, scaled_data, scaler, time_step, n_periods)

        if forecast_type == 'Day':
            forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=n_periods, freq='D')
        elif forecast_type == 'Week':
            forecast_dates = pd.date_range(start=data.index[-1] + timedelta(weeks=1), periods=n_periods, freq='W')
        elif forecast_type == 'Month':
            forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=30), periods=n_periods, freq='M')

        forecast_df = pd.DataFrame(forecasted_prices, index=forecast_dates, columns=['Forecasted price'])

        st.subheader(f'Forecasted {ticker} Prices')
        st.write(forecast_df)

        st.subheader(f'{ticker} Price Forecasting Plot')
        plt.figure(figsize=(12, 6))
        plt.plot(data[-100:], label=f'Original {ticker} Prices')
        plt.plot(forecast_df, label=f'Forecasted {ticker} Prices')
        plt.legend()
        st.pyplot(plt)

        # st.write('Forecast DataFrame:')
        # st.write(forecast_df)

    #if sub_menu =='Evaluation':
        st.write('Model Evaluation')
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert predictions to original scale
        train_predict = scaler.inverse_transform(train_predict)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        test_predict = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))


        # Calculate RMSE
        train_rmse = np.sqrt(np.mean(np.square(y_train_inv - train_predict)))
        test_rmse = np.sqrt(np.mean(np.square(y_test_inv - test_predict)))

        st.write(f'Train RMSE: {train_rmse}')
        st.write(f'Test RMSE: {test_rmse}')


         
    

   
    


#         # Plot results
# plt.figure(figsize=(16,8))
# plt.plot(data.index[seq_length:train_size+seq_length], y_train_inv, label='Actual (Train)')
# plt.plot(data.index[seq_length:train_size+seq_length], train_predict, label='Predicted (Train)')
# plt.plot(data.index[train_size+seq_length:], y_test_inv, label='Actual (Test)')
# plt.plot(data.index[train_size+seq_length:], test_predict, label='Predicted (Test)')
# plt.title(f'{ticker} Stock Price Prediction using LSTM')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()