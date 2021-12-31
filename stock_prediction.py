#issues with setting up ARIMA predictions, will need to refine further to handle datetime issues


#import
from datetime import datetime
from datetime import date
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Line that is not converging


#parameter initialization
start_date = date.today() - timedelta(days=150)
end_date = date.today()
#90 day period for the training data
data = yf.download(input("Enter the stock symbol to predict... "), start=start_date, end=end_date)

print("Data Length: ", len(data))

#arim prediction goes here
train = data['Close'][:-1].values
print("\n\n")
#print(test)
#exit()
#need to run grid search for parameters


print("Training Data and current value for comparison to predict" , data['Close'].values)
#predictions = (model_fit.predict(start=1, end=len(data)))
print("Performing Predictions for 25, 30, 35, and 50 regressions...")

predictions = []

model = ARIMA(train, order=(25,1,0)) #first hyper parameter value represents the regression amount
model_fit = model.fit()
prediction = model_fit.forecast()
predictions.append(prediction)


model = ARIMA(train, order=(30,1,0)) #first hyper parameter value represents the regression amount
model_fit = model.fit()
prediction = model_fit.forecast()
predictions.append(prediction)


model = ARIMA(train, order=(35,1,0)) #first hyper parameter value represents the regression amount
model_fit = model.fit()
prediction = model_fit.forecast()
predictions.append(prediction)


model = ARIMA(train, order=(50,1,0)) #first hyper parameter value represents the regression amount
model_fit = model.fit()
prediction = model_fit.forecast()
predictions.append(prediction)


print("Predictions for tomorrows Stock Price", predictions)
print("Average Prediction Value: ", sum(predictions)/len(predictions))


#Display data
plt.figure(figsize = (20,10))
plt.title('Stock Prices at Close for 90 day period')
#plt.plot(data['Close'])
plt.plot(data['Close'].values, 'bs')
plt.plot([len(train),len(train),len(train),len(train)], predictions, 'rD')
plt.plot(len(train), sum(predictions)/len(predictions), 'gD')
#plt.plot(prediction, 'rD')
plt.show()
