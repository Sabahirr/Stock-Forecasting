import streamlit as st
import yfinance as yf
import time
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import matplotlib.pyplot as plt
import os

icon = Image.open('icon.png')
logo = Image.open('home1.png')
homep = Image.open('home1.png')
banner1 = Image.open('ban.jpeg')
banner2 = Image.open('bann.jpg')


st.set_page_config(layout='wide', page_title="Sabahir's app", page_icon=icon)
st.markdown("<h1 style='color: #1f77b4;'>Welcome to the Stock Price Forecasting App!</h1>", unsafe_allow_html=True)
st.text('Simple Machine Learning Web Application with Streamlit')

# Sidebar Container
st.sidebar.image(image=logo)
menu = st.sidebar.selectbox('Select Page:', ['Homepage', 'Forecast', 'Evaluate'])



def get_stock_data(ticker, interval):
    stock_info = yf.Ticker(ticker)
    stock_history = stock_info.history(period="max")
    start_date = stock_history.index.min().strftime('%Y-%m-%d')
    end_date = stock_history.index.max().strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data['Close']

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
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
    while i < n_periods:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    forecasted_stock_prices = scaler.inverse_transform(lst_output)
    return forecasted_stock_prices





if menu == 'Homepage':
    
    
    st.subheader("Overview")
    st.markdown("""
        <p style='color: #ff7f0e;'>
        This application allows you to forecast stock prices using a Long Short-Term Memory (LSTM) neural network model. 
        You can either train a new model or use an existing model to make predictions.
        </p>
    """, unsafe_allow_html=True)

    st.subheader("How to Use the App")
    
    st.markdown("<h3 style='color: #2ca02c;'>1. Model Page:</h3>", unsafe_allow_html=True)
    st.markdown("""
        <ul style='color: #2ca02c;'>
        <li>Enter the stock ticker symbol (e.g., 'AAPL' for Apple Inc., 'GOOGL' for Alphabet Inc.).</li>
        <li>Select the forecast type (Day, Week, or Month).</li>
        <li>Click on 'Fetch and Forecast' to fetch the stock data and train the model.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #d62728;'>2. Evaluation Page:</h3>", unsafe_allow_html=True)
    st.markdown("""
        <ul style='color: #d62728;'>
        <li>Enter the stock ticker symbol or select from the most popular stocks.</li>
        <li>Select the forecast type (Day, Week, or Month).</li>
        <li>Click on 'Evaluate' to evaluate the model's performance.</li>
        </ul>
    """, unsafe_allow_html=True)

    # st.image(homep, use_column_width='always')
    


elif menu == 'Forecast':
   

    st.image(banner1, use_column_width='always')
    entered = st.text_input('Enter the stock ticker or select the most popular one:')
    st.warning(""" You can enter the name of any stock or cryptocurrency listed in yahoo finance as is. https://finance.yahoo.com/
                  
                """)
   
    most_popular = st.selectbox('Most popular:', ['BTC-USD', 'GC=F' , 'SI=F' , 'CL=F' , '^GSPC' , '^IXIC' , 'NVDA' , 'AAPL' , 'GOOGL' , 'TSLA', 'ETH-USD',
                                                        'BNB-USD','SOL-USD', 'XRP-USD'])
    st.info("""
                **Bitcoin** : 'BTC-USD',   **Gold** : 'GC=F' ,    **Silver** : 'SI=F' ,    **Crude Oil** : 'CL=F' ,    **S&P 500** : '^GSPC' ,   **Nasdaq** : '^IXIC', **NVIDIA**: 'NVDA' , 
          **Apple**: 'AAPL' , **Google**: 'GOOGL' , **Tesla** : 'TSLA' ,  **Ethereum** : 'ETH-USD' 

                """)
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

    if st.button('Forecast'):


        # progress_bar = st.progress(0)
        # progress_text = st.empty()

        # progress_bar.progress(10)
        # progress_text.text("Fetching stock data...")
        # data = get_stock_data(ticker, interval)
        # data = data.resample(resample_interval).ffill()

        # progress_bar.progress(30)
        # progress_text.text("Scaling data...")






        data = get_stock_data(ticker, interval)
        data = data.resample(resample_interval).ffill()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        time_step = 100
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # train_size = int(len(X) * 0.8)
        # X_train, X_test = X[0:train_size], X[train_size:len(X)]
        # y_train, y_test = y[0:train_size], y[train_size:len(y)]

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
        

         # Create a progress bar
        progress_bar = st.progress(0)

        # Define a custom callback to update progress bar
        class StreamlitProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params['epochs']
                progress_bar.progress(progress)
                #st.write(f'Epoch {epoch+1}/{self.params["epochs"]}, Loss: {logs["loss"]:.4f}, Val Loss: {logs["val_loss"]:.4f}')

        # Train the model with the custom callback
        history = model.fit(
            X, y,
            batch_size=32,
            epochs=2,  # Adjust the number of epochs as needed
            validation_split=0.2,
            callbacks=[early_stopping, StreamlitProgressCallback()],
            verbose=0  # Set verbose to 0 to prevent TensorFlow from printing to console
        )


        # history = model.fit(
        #     X_train, y_train,
        #     batch_size=32,
        #     epochs=2,
        #     validation_split=0.2,
        #     callbacks=[early_stopping],
        #     verbose=1
        # )

        # progress_bar.progress(80)
        # progress_text.text("Forecasting stock prices...")
        


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

elif menu == 'Evaluate':
    st.write('Model Evaluation')

    st.image(banner2, use_column_width='always')
    entered = st.text_input('Enter the stock ticker or select the most popular one:')
    st.warning(""" You can enter the name of any stock or cryptocurrency listed in yahoo finance as is. https://finance.yahoo.com/
                  
                """)
    most_popular = st.selectbox('Most popular:', ['BTC-USD', 'GC=F' , 'SI=F' , 'CL=F' , '^GSPC' , '^IXIC' , 'NVDA' , 'TSLA', 'ETH-USD',
                                                        'BNB-USD','SOL-USD', 'XRP-USD'])
    st.info("""
                **Bitcoin** : 'BTC-USD',   **Gold** : 'GC=F' ,    **Silver** : 'SI=F' ,    **Crude Oil** : 'CL=F' ,    **S&P 500** : '^GSPC' ,   **Nasdaq** : '^IXIC', **NVIDIA**: 'NVDA' , 
          **Tesla** : 'TSLA' ,  **Ethereum** : 'ETH-USD' 

                """)
    ticker = entered or most_popular
    forecast_type = st.selectbox('Select evaluate type:', ['Day', 'Week', 'Month'])

    if forecast_type == 'Day':
        interval = '1d'
        resample_interval = 'D'
        
    elif forecast_type == 'Week':
        interval = '1wk'
        resample_interval = 'W'
        
    elif forecast_type == 'Month':
        interval = '1mo'
        resample_interval = 'M'
      
    else:
        st.error("Invalid forecast type. Please select 'Day', 'Week', or 'Month'.")

    if st.button('Evaluate'):

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


         # Create a progress bar
        progress_bar = st.progress(0)

        # Define a custom callback to update progress bar
        class StreamlitProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params['epochs']
                progress_bar.progress(progress)
                st.write(f'Epoch {epoch+1}/{self.params["epochs"]}, Loss: {logs["loss"]:.4f}, Val Loss: {logs["val_loss"]:.4f}')

        # Train the model with the custom callback
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,  # Adjust the number of epochs as needed
            validation_split=0.2,
            callbacks=[early_stopping, StreamlitProgressCallback()],
            verbose=0  # Set verbose to 0 to prevent TensorFlow from printing to console
        )



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

        st.info(f'Train RMSE: {train_rmse}')
        st.info(f'Test RMSE: {test_rmse}')

        st.warning("""RMSE is the average of the errors the model can make""")

         # Plot results
        plt.figure(figsize=(16, 8))
        plt.plot(data.index[time_step:train_size+time_step], y_train_inv, label='Actual (Train)')
        plt.plot(data.index[time_step:train_size+time_step], train_predict, label='Predicted (Train)')
        plt.plot(data.index[train_size+time_step:train_size+time_step+len(y_test_inv)], y_test_inv, label='Actual (Test)')
        plt.plot(data.index[train_size+time_step:train_size+time_step+len(test_predict)], test_predict, label='Predicted (Test)')
        plt.title(f'{ticker} Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)


        




        
    
