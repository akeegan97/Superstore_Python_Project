
#Given a set dataset of key variables from a fictional superstore.
# file path of data : "C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\Sample - Superstore.csv"
#Dataset given by Kaggle.com open datasets

from statistics import mode
from xml.parsers.expat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#import data set
super_store_data  = r"C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\Sample - Superstore.csv"
raw_data = pd.read_csv(super_store_data, encoding='Latin1')

""" print(raw_data.head()) """



raw_data.columns = raw_data.columns.str.replace(' ','')
raw_data.columns = raw_data.columns.str.replace('-','')

#Changing standard date to PD DATETIME
#datetime in format YYYY-MM-DD
raw_data['OrderDate'] = pd.to_datetime(raw_data['OrderDate'], infer_datetime_format= True)
raw_data['ShipDate'] = pd.to_datetime(raw_data['ShipDate'], infer_datetime_format=True)


#Making 3 new Dataframes filtered raw_data by Main Category


furniture_df = raw_data.loc[raw_data['Category'] == 'Furniture']
office_df = raw_data.loc[raw_data['Category'] == 'Office Supplies']
tech_df = raw_data.loc[raw_data['Category']=='Technology'] 

#Function to filter Main DFS by year

def yearly(dataframe, start, end):
    start_date = pd.Timestamp(start, 1, 1)
    end_date = pd.Timestamp(end, 12, 31)
    newdf = dataframe[(dataframe['OrderDate'] >= start_date) & (dataframe['OrderDate'] <= end_date)]
    return newdf

##Function to take the 3 filtered DFS, and return a DF that is multiindexed with columns [Profit, State, Region, Customers]

def getinfo(df,y1,y2,y3,y4,title):
    year1 = pd.Timestamp(y1,1,1)
    year2 = pd.Timestamp(y2,1,1)
    year3 = pd.Timestamp(y3,1,1)
    year4 = pd.Timestamp(y4,1,1)
    year5 = pd.Timestamp(y4,12,31)

    df1 = df[(df['OrderDate'] >= year1) & (df['OrderDate'] < year2)]

    profit = df1['Profit'].sum()
    sales = df1['Sales'].sum()
    customers = df1['CustomerID'].nunique()
    region = mode(list(df1['Region']))
    state = mode(list(df1['State']))
    quantity = df1['Quantity'].sum()

    dataframe = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y1],
        'Category' : [title],
        'Quantity' : [quantity]
    }
    
    dataframe = pd.DataFrame(dataframe)
    

    df2 = df[(df['OrderDate'] >= year2) & (df['OrderDate'] < year3)]

    profit = df2['Profit'].sum()
    sales = df2['Sales'].sum()
    customers = df2['CustomerID'].nunique()
    region = mode(list(df2['Region']))
    state = mode(list(df2['State']))
    quantity = df2['Quantity'].sum()

    dataframe2 = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y2],
        'Category' : [title],
        'Quantity' : [quantity]
    }
    

    dataframe2 = pd.DataFrame(dataframe2)
    

    df3 = df[(df['OrderDate'] >= year3) & (df['OrderDate'] <= year4)]

    profit = df3['Profit'].sum()
    customers = df3['CustomerID'].nunique()
    region = mode(list(df3['Region']))
    state = mode(list(df3['State']))
    sales = df3['Sales'].sum()
    quantity = df3['Quantity'].sum()

    dataframe3 = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y3],
        'Category' : [title],
        'Quantity' : [quantity]
    }
    
    dataframe3 = pd.DataFrame(dataframe3)

    df4 = df[(df['OrderDate'] >= year4) & (df['OrderDate'] < year5)]

    profit = df4['Profit'].sum()
    customers = df4['CustomerID'].nunique()
    region = mode(list(df4['Region']))
    state = mode(list(df4['State']))
    sales = df4['Sales'].sum()
    quantity = df4['Quantity'].sum()

    dataframe4 = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y4],
        'Category' : [title],
        'Quantity' : [quantity]
    }
    
    dataframe4 = pd.DataFrame(dataframe4)
   
    dfs_to_combine = (dataframe,dataframe2,dataframe3,dataframe4)

    DFO = pd.concat(dfs_to_combine)
    
    DFO = DFO.set_index(['Category', 'Year'])

    return DFO




furniture_out = getinfo(furniture_df,2014,2015,2016,2017,'Furniture')
tech_out = getinfo(tech_df,2014,2015,2016,2017,'Tech')
office_out = getinfo(office_df,2014,2015,2016,2017,'Office')

###combining the three seperate "getinfo" dfs into one main that consists of a multiindexed DF (year, month) and columns Profit, Sales, State, Region, 

out_DFS = (furniture_out,tech_out,office_out)

big_DFO = pd.concat(out_DFS)


##Define Function that takes Big_DFO and gives which category had the highest Profit, Sales, or number of Customers for a given Year


def answers(df,year,data):
    df = pd.DataFrame(df)
    df1 = df[df.index.isin([year],level=1)]
    df_out = df1.loc[df1[data].idxmax()]
    df_out = pd.DataFrame(df_out)
    df_out = df_out.T
    df_out = df_out[data]
    df_out =pd.DataFrame(df_out)
    return df_out






###Forcasting
#Forecasting the number of units sold per each business category tech, furniture, and office supplies



def group(dataframe):
    df1 =dataframe.copy()

    df1['Year'] = dataframe['OrderDate'].dt.to_period('Y')
    df1['Month'] = dataframe['OrderDate'].dt.to_period('M')

    df = df1.groupby(['Year','Month'])[['Sales','Quantity','Profit']].sum()
    df2 = df1.groupby(['Year','Month'])[['CustomerID']].nunique()
    df2 = df2.rename(columns={'CustomerID':'Customers'})
    df3 = pd.merge(df,df2,left_index=True,right_index=True)


    return df3


###Getting all categories grouped and ready to be ran through following functions

##Furniture
furniture_grouped = group(furniture_df)
sales_f = furniture_grouped['Sales']
sales_f = sales_f.droplevel(level=0)
profit_f = furniture_grouped['Profit']
profit_f = profit_f.droplevel(level=0)
quantity_f = furniture_grouped['Quantity']
quantity_f = quantity_f.droplevel(level=0)
sales_f.index=sales_f.index.to_timestamp()
profit_f.index=profit_f.index.to_timestamp()
quantity_f.index=quantity_f.index.to_timestamp()
##Office
office_grouped = group(office_df)
sales_o = office_grouped['Sales']
sales_o = sales_o.droplevel(level=0)
profit_o = office_grouped['Profit']
profit_o = profit_o.droplevel(level=0)
quantity_o = office_grouped['Quantity']
quantity_o = quantity_o.droplevel(level=0)
sales_o.index=sales_o.index.to_timestamp()
profit_o.index=profit_o.index.to_timestamp()
quantity_o.index=quantity_o.index.to_timestamp()
##Tech
tech_grouped = group(tech_df)
sales_t = tech_grouped['Sales']
sales_t = sales_t.droplevel(level=0)
profit_t = tech_grouped['Profit']
profit_t = profit_t.droplevel(level=0)
quantity_t = tech_grouped['Quantity']
quantity_t = quantity_t.droplevel(level=0)
sales_t.index=sales_t.index.to_timestamp()
profit_t.index=profit_t.index.to_timestamp()
quantity_t.index=quantity_t.index.to_timestamp()
###Start Timeseries analysis of the categories

import statsmodels.api as sm

###Defined function to see the seasonality breakdown of each category
def seasonal_decompose(y):
    decomposition = sm.tsa.seasonal_decompose(y,model='additive',)
    fig =decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


###Starting the SARIMA model for analysis and forecasting
##Testing stationarity of data 

def test_stationarity(timeseries, title):
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()


    fig, ax = plt.subplots(figsize = (16,4))
    ax.plot(timeseries,label = title)
    ax.plot(rolmean, label = 'Rolling Mean')
    ax.plot(rolstd,label = 'Rolling STD (x10)')
    ax.legend()
    plt.show()
    

pd.options.display.float_format = '{:.8f}'.format


##Testing Stationarity using the Augmented Dickery-Fuller Test

from statsmodels.tsa.stattools import adfuller

##Defining function to use the ADF test of stationarity

def ADF_test(timeseries, dataDesc):
    print('> Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - the data is {} stationary with {}% confidence'.format(k,v,'not' if v < dftest[0] else '', 100-int(k[:-1])))


###getting data split for the to train data set, data to compare model to actual and getting length of forecast

###Furniture
furniture_sales_to_train = sales_f[:'2016-12-01']
furniture_sales_to_validate = sales_f[:'2017-12-01']
predict_date = len(furniture_sales_to_train)-len(furniture_sales_to_validate)
furniture_profit_to_train = profit_f[:'2016-12-01'] 
furniture_profit_to_validate = profit_f[:'2017-12-01']
furniture_quantity_to_train = quantity_f[:'2016-12-01']
furniture_quantity_to_validate = quantity_f[:'2017-12-01']
###Office
office_sales_to_train = sales_o[:'2016-12-01']
office_sales_to_validate = sales_o[:'2017-12-01']
office_profit_to_train = profit_o[:'2016-12-01']
office_profit_to_validate = profit_o[:'2017-12-01']
office_quantity_to_train = quantity_o[:'2016-12-01']
office_quantity_to_validate = quantity_o[:'2017-12-01']
###Tech
tech_sales_to_train = sales_t[:'2016-12-01']
tech_sales_to_validate = sales_t[:'2017-12-01']
tech_profit_to_train = profit_t[:'2016-12-01']
tech_profitto_validate = profit_t[:'2017-12-01']
tech_quantity_to_train = quantity_t[:'2016-12-01']
tech_quantity_to_validate = quantity_t[:'2017-12-01']





###Implementing a grid search approach to find the optimal pramater values for the sarima
import itertools


def sarima_grid_search(y, seasonal_period):
    p = d = q  =range(0,2)
    pdq = list(itertools.product(p,d,q))
    seasonal_pdq = [(x[0],x[1],x[2], seasonal_period) for x in list(itertools.product(p,d,q))]

    mini = float('+inf')


    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try: 
                mod = sm.tsa.statespace.SARIMAX(y,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False)
                results = mod.fit()
                if results.aic <mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal
            except:
                continue
    print('the set of params with min AIC is SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))

###Function that returns the actual model for the data set
def sarima_eva(y, order, seasonal_order, seasonal_period, pred_date, y_to_test):
    mod = sm.tsa.statespace.SARIMAX(
        y, 
    order=order,
    seasonal_order=seasonal_order,
    enforce_invertibility=False,
    )
    results = mod.fit()
    print(results.summary().tables[1])

    pred = results.get_prediction(start=pd.to_datetime(pred_date),dynamic = False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test)**2).mean()
    print('The Root Mean Squared Error of Sarima w/ seasonal_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse),2)))


    pred_dynamic = results.get_prediction(start = pd.to_datetime(pred_date), dynamic = True, full_results = True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test)**2).mean()
    print('The Root Mean Square Error of Sarima w/season_length = {} and dynamic = TRUE {}'.format(seasonal_period,round(np.sqrt(mse_dynamic),2)))


    return(results)

#function for evaluation plots:

def sarima_eva_plots(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
    mod = sm.tsa.statespace.SARIMAX(
        y, 
    order=order,
    seasonal_order=seasonal_order,
    enforce_invertibility=False,
    )
    results = mod.fit()
    results.plot_diagnostics(figsize=(16,8))
    plt.show()
    pred = results.get_prediction(start=pd.to_datetime(pred_date),dynamic = False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test)**2).mean()
    ax = y.plot(label = 'Observed')
    y_forecasted.plot(ax=ax, label = 'One-Step ahead Forecast', alpha = .7, figsize = (14,7))
    ax.fill_between(pred_ci.index,
    pred_ci.iloc[:,0],
    pred_ci.iloc[:,1], color = 'k', alpha = .2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    plt.legend()
    plt.show()
    pred_dynamic = results.get_prediction(start = pd.to_datetime(pred_date), dynamic = True, full_results = True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test)**2).mean()
    
    ax = y.plot(label = 'Observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize = (14,7))
    ax.fill_between(pred_dynamic_ci.index,
    pred_dynamic_ci.iloc[:,0],
    pred_dynamic_ci.iloc[:,1], color = 'k', alpha = .2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    plt.legend()
    plt.show()

###Define Function that makes the forecasted data

def forecast(model, predict_steps, y):
    pred_uc = model.get_forecast(steps = predict_steps)
    pred_ci = pred_uc.conf_int()
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date', 'Predicted_Mean']
    pci = pred_ci.reset_index()
    pci.columns = ['Date', 'Lower Bound', 'Upper Bound']
    final_table = pm.join(pci.set_index('Date'), on = 'Date')
    final_table = pd.DataFrame(final_table)

    return(final_table)


#Function that displays the forecasted data:

def forecasted_plot(model,predict_steps,y):
    pred_uc = model.get_forecast(steps = predict_steps)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label = 'Observed', figsize= (14,7))
    pred_uc.predicted_mean.plot(ax=ax, label = "Forecast")
    ax.fill_between(pred_ci.index,
    pred_ci.iloc[:,0],
    pred_ci.iloc[:,1], color = 'k', alpha = .25)
    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)
    plt.legend()
    plt.show()





###Starting Forecasting for Quantity sold Category Furniture###

#ADF_test(quantity_f, 'Quantity')

###OUTPUT of Above ADF Test
#Test statistic = -3.677
#P-value = 0.004
#Critical values :
#        1%: -3.5778480370438146 - the data is  stationary with 99% confidence
#        5%: -2.925338105429433 - the data is  stationary with 95% confidence
#        10%: -2.6007735310095064 - the data is  stationary with 90% confidence

#seasonal_decompose(quantity_f)
#sarima_grid_search(quantity_f,12)

##OUTPUT of grid search for furniture quantity sold
# the set of params with min AIC is SARIMA(0, 1, 1)x(1, 1, 1, 12) - AIC:208.9779784268398##

model_quantity_furniture = sarima_eva(quantity_f,(0, 1, 1),(1, 1, 1, 12),12,'2016-12-01',furniture_quantity_to_validate)

##OUTPUT of Sarima_EVA
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#ma.L1         -0.9353      0.216     -4.336      0.000      -1.358      -0.512
#ar.S.L12      -0.3194      0.861     -0.371      0.711      -2.008       1.369
#ma.S.L12      -0.0465      1.094     -0.042      0.966      -2.192       2.099
#sigma2       926.2199    312.831      2.961      0.003     313.082    1539.358
#==============================================================================
#The Root Mean Squared Error of Sarima w/ seasonal_length=12 and dynamic = False 31.99
#The Root Mean Square Error of Sarima w/season_length = 12 and dynamic = TRUE 34.45


###Running the plotting of the models and actuals

#sarima_eva_plots(quantity_f,(0, 1, 1),(1, 1, 1, 12),12,'2016-12-01',furniture_quantity_to_validate)
#forecasted_plot(model_quantity_furniture,12,quantity_f)

##Getting the actual dataframe of forecasted outputs for furniture quantity
out_f = forecast(model_quantity_furniture,12,quantity_f)


###Starting Forecasting for Quantity sold Category Office###

#ADF_test(quantity_o, 'Quantity')

##OUTPUT of Above ADF test
#Test statistic = -3.599
#P-value = 0.006
#Critical values :
#        1%: -3.5778480370438146 - the data is  stationary with 99% confidence
#        5%: -2.925338105429433 - the data is  stationary with 95% confidence
#        10%: -2.6007735310095064 - the data is  stationary with 90% confidence

#seasonal_decompose(quantity_o)

#sarima_grid_search(quantity_o,12)

# OUTPUT of Grid Search for Office Quantity Sold
#the set of params with min AIC is SARIMA(0, 1, 1)x(0, 1, 1, 12) - AIC:252.39851522536642

model_quantity_office = sarima_eva(quantity_o,(0, 1, 1),(0, 1, 1, 12),12,'2016-12-01',office_quantity_to_validate)
out_o = forecast(model_quantity_office,12,quantity_o)

###OUTPUT of Sarima eva 
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#ma.L1         -0.6275      0.169     -3.720      0.000      -0.958      -0.297
#ma.S.L12      -0.4162      0.241     -1.726      0.084      -0.889       0.056
#sigma2      5615.6716   2041.893      2.750      0.006    1613.636    9617.708
#==============================================================================
#The Root Mean Squared Error of Sarima w/ seasonal_length=12 and dynamic = False 89.03
#The Root Mean Square Error of Sarima w/season_length = 12 and dynamic = TRUE 131.91

#forecasted_plot(model_quantity_office,12,quantity_o)
#sarima_eva_plots(quantity_o,(0, 1, 1),(0, 1, 1, 12),12,'2016-12-01',office_quantity_to_validate)


###Quantity Sold for Tech Category

ADF_test(quantity_t,"Quantity")

###OUTPUT of ADF test
#Test statistic = -4.119
#P-value = 0.001
#Critical values :
#        1%: -3.5778480370438146 - the data is  stationary with 99% confidence
#        5%: -2.925338105429433 - the data is  stationary with 95% confidence
#        10%: -2.6007735310095064 - the data is  stationary with 90% confidence

###Decomposition of Seasonality for Tech Quantity
#seasonal_decompose(quantity_t)

###Sarima Paramater Grid Search
#sarima_grid_search(quantity_f,12)

###OUTPUT of grid search
#the set of params with min AIC is SARIMA(0, 1, 1)x(1, 1, 1, 12) - AIC:208.9779784268398

###Sarima model

model_t = sarima_eva(quantity_t,(0, 1, 1),(1, 1, 1, 12),12,'2016-12-01',tech_quantity_to_validate)

###OUTPUT of Sarima eva
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#ma.L1         -1.2436      0.161     -7.745      0.000      -1.558      -0.929
#ar.S.L12       0.5792      1.622      0.357      0.721      -2.601       3.759
#ma.S.L12      -1.0002   2290.795     -0.000      1.000   -4490.875    4488.875
#sigma2       819.4953   1.88e+06      0.000      1.000   -3.68e+06    3.68e+06
#==============================================================================
#The Root Mean Squared Error of Sarima w/ seasonal_length=12 and dynamic = False 46.32
#The Root Mean Square Error of Sarima w/season_length = 12 and dynamic = TRUE 54.87


#forecasted_plot(model_t,12,quantity_t)


out_t = forecast(model_t,12,quantity_t)



####out_t,out_f,out_o are DF that contain the forecasted values based on the Sarima model

###want to combine the out_xyz with the xyz_quantity DF in order to plot the forecasted amounts
###generalized average profit per unit sold and sales per unit sold for each of the three categories



##Average profit/sales per unit
#Furniture
F_average_profit_per_unit = round(furniture_grouped['Profit'].sum()/furniture_grouped['Quantity'].sum(),2)
F_average_sales_per_unit = round(furniture_grouped['Sales'].sum()/furniture_grouped['Quantity'].sum(),2)
#Tech
T_average_profit_per_unit = round(tech_grouped['Profit'].sum()/tech_grouped['Quantity'].sum(),2)
T_average_sales_per_unit = round(tech_grouped['Sales'].sum()/tech_grouped['Quantity'].sum(),2)
#Office
O_average_profit_per_unit = round(office_grouped['Profit'].sum()/office_grouped['Quantity'].sum(),2)
O_average_sales_per_unit = round(office_grouped['Sales'].sum()/office_grouped['Quantity'].sum(),2)

##Getting the Mean forecast of each category
furniture_forecast_Q = out_f['Predicted_Mean'] 
tech_forecast_Q = out_t['Predicted_Mean']
office_forecast_Q = out_o['Predicted_Mean']


out_t_to_concat = out_t.copy(deep=True)
out_t_to_concat['Date'] = pd.to_datetime(out_t['Date'],infer_datetime_format= True)
out_t_to_concat['Year'] = out_t_to_concat['Date'].dt.to_period('Y')
out_t_to_concat['Month'] =out_t_to_concat['Date'].dt.to_period('M')
out_t_to_concat = out_t_to_concat.drop(columns='Date')
out_t_to_concat = out_t_to_concat.set_index(['Year','Month'])
out_t_to_concat = out_t_to_concat.rename(columns={'Predicted_Mean':'Quantity'})


out_t_profit = np.array(out_t_to_concat['Quantity'] * T_average_profit_per_unit)
out_t_sales = np.array(out_t_to_concat['Quantity']*T_average_sales_per_unit)
out_t_quantity = np.array(out_t_to_concat['Quantity'])
index_out = pd.to_datetime(out_t['Date'],infer_datetime_format= True)
out_t_df = pd.DataFrame(
{
        'Date' : index_out,
        'Sales':out_t_sales,
        'Profit':out_t_profit,
        'Quantity':out_t_quantity
}
)


###Function to make the concat dfs 

def concat(df1,ppq1,spq1):
    df1 = pd.DataFrame(df1)
    df_out_1 = df1.copy(deep=True)
    df_out_1['Date'] = pd.to_datetime(df1['Date'],infer_datetime_format=True)
    df_out_1 = df_out_1.rename(columns={'Predicted_Mean':'Quantity'})
    index_out = df_out_1['Date'] = pd.to_datetime(df1['Date'],infer_datetime_format=True)
    out_p_1 = np.array(df_out_1['Quantity']*ppq1)
    out_s_1 = np.array(df_out_1['Quantity']*spq1)
    out_q_1 = np.array(df_out_1['Quantity'])
    df_out_1['Year'] = df_out_1['Date'].dt.to_period('Y')
    df_out_1['Month'] = df_out_1['Date'].dt.to_period('M')
    df_out_1 = df_out_1.drop(columns='Date')
    df_out_1 = df_out_1.set_index(['Year','Month'])
    df = pd.DataFrame(
    {
        'Date' : index_out,
        'Sales':out_s_1,
        'Quantity':out_q_1,
        'Profit' :out_p_1
    }
    )
    df['Year'] = df['Date'].dt.to_period('Y')
    df['Month'] = df['Date'].dt.to_period('M')
    df = df.drop(columns='Date')
    df = df.set_index(['Year','Month'])

    return df

forecast_f = concat(out_f,F_average_profit_per_unit,F_average_sales_per_unit)
forecast_t = concat(out_t,T_average_profit_per_unit,T_average_sales_per_unit)
forecast_o = concat(out_o,O_average_profit_per_unit,O_average_sales_per_unit)

furniture_to_concat = furniture_grouped.drop(columns='Customers')
tech_to_concat = tech_grouped.drop(columns='Customers')
office_to_concat = office_grouped.drop(columns='Customers')



dfs_furniture = (furniture_to_concat,forecast_f)
dfs_tech = (tech_to_concat,forecast_t)
dfs_office = (office_to_concat,forecast_o)
total_furniture = pd.DataFrame(pd.concat(dfs_furniture))
total_office = pd.DataFrame(pd.concat(dfs_office))
total_tech = pd.DataFrame(pd.concat(dfs_tech))



####Plotting
###Getting the x,y arrays necessary to plot

sales_plot_f=np.array(total_furniture['Sales'].values)
sales_plot_t=np.array(total_tech['Sales'].values)
sales_plot_o=np.array(total_office['Sales'].values)

profit_plot_f=np.array(total_furniture['Profit'].values)
profit_plot_t=np.array(total_tech['Profit'].values)
profit_plot_o=np.array(total_office['Profit'].values)

quantity_plot_f=np.array(total_furniture['Quantity'])
quantity_plot_t = np.array(total_tech['Quantity'])
quantity_plot_o=np.array(total_office['Quantity'])

plotting_timeseries = total_office.droplevel(level=0)
plotting_timeseries = total_office.reset_index()
plotting_timeseries['Month'] = plotting_timeseries['Month'].dt.to_timestamp('s').dt.strftime('%Y-%m')
plot_ts = plotting_timeseries['Month']


###Creating function to call that will display the forecasted profit,sales,quantity 
def final(plot):
    x_ticks = 12
    if plot == 'Yes' :
        plt.figure(figsize=(18,7))
        plt.plot(plot_ts,quantity_plot_f,'r',label = 'Furniture Quantity')
        plt.plot(plot_ts,quantity_plot_o,'g',label='Office Quantity')
        plt.plot(plot_ts,quantity_plot_t,'y',label='Tech Quantity')
        plt.xticks(range(0,len(plot_ts),x_ticks),plot_ts[::x_ticks],rotation = 45)
        plt.axvline('2018-01',label='Forecast')
        plt.axvspan('2018-01','2018-12',facecolor = 'blue',alpha = .25)
        plt.title('Historical and Forecasted \n Units Sold')
        plt.legend()
        plt.savefig(r'C:\Users\Andrew\Desktop\Python_Projects\Website\images\profit_forecast.jpg', bbox_inches = 'tight')
        """ plt.show() """
        plt.figure(figsize=(18,7))
        plt.plot(plot_ts,profit_plot_f,'r',label='Furniture Profit')
        plt.plot(plot_ts,profit_plot_o,'g',label = 'Office Profit')
        plt.plot(plot_ts,profit_plot_t,'y',label='Tech Profit')
        plt.xticks(range(0,len(plot_ts),x_ticks),plot_ts[::x_ticks],rotation = 45)
        plt.axvline('2018-01',label='Forecast')
        plt.axvspan('2018-01','2018-12',facecolor = 'blue',alpha = .25)
        plt.title('Historical and Forecasted Profit \n Based on Average Profit per Unit Sold')
        plt.legend()
        plt.savefig(r'C:\Users\Andrew\Desktop\Python_Projects\Website\images\units_sold_forecast.jpg', bbox_inches = 'tight')
        plt.show()

    else:
        print('No')


final('No')

print(answers(big_DFO,'2016','Profit'))

print(big_DFO)







