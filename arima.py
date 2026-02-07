import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings 
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import seaborn as sns
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf
import csv

def process(path,sname,nop):
    
    df_w = pd.read_csv(path, index_col='YEAR',parse_dates = True)
    df_w=df_w[df_w['SUBDIVISION'] == sname]
    df_w = df_w[df_w.index>'1901']
    df_w.head()

    df_w  = df_w[['ANNUAL']]
    plt.plot(df_w['ANNUAL'],label = 'Rainfall')
    plt.title('ANNUAL 1901 to 2017',fontsize=20)
    plt.xlabel('Years', fontsize=15)
    plt.ylabel('ANNUAL Rainfall',fontsize=15)
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(16.5, 4.5)
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    fig.savefig("results/livechart.png")


    decomposition = seasonal_decompose(df_w,model='additive',two_sided = False,freq =52 )
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    resid = decomposition.resid
    plt.plot(trend,label='Rainfall')
    plt.xlabel('Year', fontsize=15)
    plt.title('Trend',fontsize=20)
    plt.ylabel('Rainfall',fontsize=15)
    plt.xticks(rotation=90)
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(16.5, 4.5)
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    fig.savefig("results/Trend.png")

    # plt.plot(seasonal,label='Seasonality')
    # plt.xlabel('Years', fontsize=15)
    # plt.ylabel('Rainfall', fontsize=15)
    # plt.title('Seasonal',fontsize=20)
    # plt.xticks(rotation=90)
    # plt.legend(loc='upper left')
    # fig = plt.gcf()
    # plt.pause(5)
    # plt.show(block=False)
    # plt.close()
    # fig.set_size_inches(15, 4.5)
    # fig.savefig("results/Seasonality.png")


    ###### Adjusting Outliers #######
    wnd = 20

    df_w['RollingStd'] = df_w['ANNUAL'].rolling(window=wnd).std()
    df_w['Rollingmean'] = df_w['ANNUAL'].rolling(window=wnd).mean()

    st= df_w['RollingStd'][wnd]
    mn= df_w['Rollingmean'][wnd]


    for i in range(wnd+1,len(df_w)):
        if df_w['RollingStd'][i]-st > st:
            df_w['RollingStd'][i] = st*1.96
            df_w['CO'][i] = mn+st*1.96
            if mn > df_w['Rollingmean'][i]:
                df_w['Rollingmean'][i] = mn-st       
            else: 
                df_w['Rollingmean'][i] = mn+st
            st = df_w['RollingStd'][i]
            mn = df_w['Rollingmean'][i]
        else:
            st = df_w['RollingStd'][i]
            mn = df_w['Rollingmean'][i]

    plt.plot(df_w['Rollingmean'],label='Rolling Mean')
    plt.plot(df_w['ANNUAL'][wnd:],label='Rainfall')
    plt.xlabel('Years', fontsize=15)
    plt.ylabel('Rainfall', fontsize=15)
    plt.title('Rolling Stats',fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(15, 4.5)
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    fig.savefig("results/Rolling Stats.png")

    plt.plot(df_w['RollingStd'],label='Rolling STD')
    plt.xlabel('Years', fontsize=15)
    plt.ylabel('Rainfall', fontsize=15)
    plt.title('Rolling STD',fontsize=20)
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(15, 4.5)
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    fig.savefig("results/Rolling STD.png")


    ############## Revenue Time series ACF and PACF Charts ####################

    df_w = df_w[['ANNUAL']]
    lag_acf = acf(df_w, nlags=20)
    lag_pacf = pacf(df_w, nlags=20, method='ols')




    #################### Looking at charts above we can create a differenced AR model of order 1  ###################
    ############### Run SARIMA Model ###################
    train = df_w['ANNUAL'][0:-10]
    test = df_w['ANNUAL'][len(train):]
    print('Train')
    print (train)
    print ('Test')
    print(test)

    p =1
    d= 0
    q = 0
    pp = 0
    dd = 1
    qq = 0
    z = 52
    aic = 'null'

    amape = 99
    af = []

    try:
        model = smt.SARIMAX(train.asfreq(freq='AS'), exog=None, order=(p, d, q), seasonal_order=(pp,dd,qq,z),trend = 'n').fit()
        aic = model.aic
        aic = round(aic,2)
        pred = model.get_forecast(len(test))
        fcst = pred.predicted_mean
        fcst.index = test.index
        mapelist = []
        for i in range(len(fcst)):
                        mapelist.insert(i, (np.absolute(test[i] - fcst[i])) / test[i])
        mape = np.mean(mapelist) * 100
        mape = round(mape,2)
    except:
        mape = 9999
        pass

    amape = mape
    sap  = p
    sad = d
    saq = q
    app = pp
    add = dd
    aqq = qq
    az = z
    af= fcst
##    mse = mean_squared_error(test, af)
##    rmse = np.sqrt(mse)
##    rmse = round(rmse,1)

    plt.plot(train)
    plt.plot(test,label='Actual')
    plt.plot(af,label='Predicted')
    fig = plt.gcf()
    fig.set_size_inches(15, 5.5)
    plt.title("Existing Prediction of "+sname ,fontsize=20)
    plt.ylabel('Rain Fall',fontsize = 15)
    plt.legend(loc='upper left')
    plt.xlabel('Years', fontsize=15)
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    fig.savefig("results/Previous.png")


    model = smt.SARIMAX(df_w.asfreq(freq='AS'), exog=None, order=(sap, sad, saq), seasonal_order=(app,add,aqq,az)).fit()
    pred = model.get_forecast(int(nop))
    cf = pred.conf_int(alpha=0.05)
    ax = df_w.plot(label='observed', figsize=(16.5, 5.5))
    pred.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(cf.index, cf.iloc[:, 0],cf.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Years',fontsize = 15)
    ax.set_ylabel('Rain Fall',fontsize = 15)
    plt.legend(loc='upper left')
    plt.title("Forecasts Result of"+sname+" from "+str(cf.index[0]).split(' ')[0]+" to "+str(cf.index[-1]).split(' ')[0],fontsize = 20)
    fig = plt.gcf()
    fig.set_size_inches(15, 5.5)
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    fig.savefig("results/Forecast.png")

    print(pred.predicted_mean)
    fcst = pred.conf_int(alpha=0.05)
    fcst['Forecast'] = pred.predicted_mean
    fcst = fcst.round(1)
    forecast = pd.DataFrame()
    forecast['Lower Rainfall'] = fcst.apply(lambda x: "{:,}".format(x['lower ANNUAL']), axis=1)
    forecast['Upper Rainfall'] = fcst.apply(lambda x: "{:,}".format(x['upper ANNUAL']), axis=1)
    forecast['Forecast Rainfall'] = fcst.apply(lambda x: "{:,}".format(x['Forecast']), axis=1)
    print(forecast)
    forecast.to_csv("results/Forecastresult"+sname+".csv", index=False, header=False)

#process("data.csv","ANDAMAN & NICOBAR ISLANDS")
