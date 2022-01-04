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


def run_prediction(training_data, hyper_parameters):
    model = ARIMA(training_data, order=hyper_parameters) #first hyper parameter value represents the regression amount
    model_fit = model.fit()
    prediction = model_fit.forecast()
    return(prediction)




#parameter initialization
start_date = date.today() - timedelta(days=150)
end_date = date.today() - timedelta(days=1)
#90 day period for the training data
data = yf.download(input("Enter the stock symbol to predict... "), start=start_date, end=end_date)

print("Data Length: ", len(data))

#arim prediction goes here
train = data['Close'].values
print("\n\n")
#print(test)
#exit()
#need to run grid search for parameters


print("Training Data and current value for comparison to predict" , data['Close'].values)
#predictions = (model_fit.predict(start=1, end=len(data)))
print("Performing Predictions for 25, 30, 35, and 50 regressions...")

predictions = []
regression_values = [25,30,35,50]

import multiprocessing as mp


pool = mp.Pool(mp.cpu_count())
predictions = [pool.apply(run_prediction, args=(train, (regress, 1, 0))) for regress in regression_values]
pool.close()

print("Predictions for Today's Clsoing Stock Price", predictions)
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
