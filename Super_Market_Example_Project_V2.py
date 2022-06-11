
#Given a set dataset of key variables from a fictional superstore.
# file path of data : "C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\Sample - Superstore.csv"
#Dataset given by Kaggle.com open datasets

from calendar import month
from statistics import mode
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


#breaking DF into the the three main categories


furniture_df = raw_data.loc[raw_data['Category'] == 'Furniture']
office_df = raw_data.loc[raw_data['Category'] == 'Office Supplies']
tech_df = raw_data.loc[raw_data['Category']=='Technology'] 

#Function to filter Main DFS by year

def yearly(dataframe, start, end):
    start_date = pd.Timestamp(start, 1, 1)
    end_date = pd.Timestamp(end, 12, 31)
    newdf = dataframe[(dataframe['OrderDate'] >= start_date) & (dataframe['OrderDate'] <= end_date)]
    return newdf


#grouping sales,profit,quantity by months for a length 12 and width 3 df 

""" def group(dataframe):
    df = dataframe.groupby(dataframe.OrderDate.dt.month)[['Profit', 'Sales','Quantity']].sum()

    df['Months'] = 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
    return df
 """


#function to take raw_data filter yearly, filter category, filter [profit,state,region,customerID] return df with ex.

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

    dataframe = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y1],
        'Category' : [title]
    }
    
    dataframe = pd.DataFrame(dataframe)
    

    df2 = df[(df['OrderDate'] >= year2) & (df['OrderDate'] < year3)]

    profit = df2['Profit'].sum()
    sales = df2['Sales'].sum()
    customers = df2['CustomerID'].nunique()
    region = mode(list(df2['Region']))
    state = mode(list(df2['State']))

    dataframe2 = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y2],
        'Category' : [title]
    }
    

    dataframe2 = pd.DataFrame(dataframe2)
    

    df3 = df[(df['OrderDate'] >= year3) & (df['OrderDate'] <= year4)]

    profit = df3['Profit'].sum()
    customers = df3['CustomerID'].nunique()
    region = mode(list(df3['Region']))
    state = mode(list(df3['State']))
    sales = df3['Sales'].sum()

    dataframe3 = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y3],
        'Category' : [title]
    }
    
    dataframe3 = pd.DataFrame(dataframe3)

    df4 = df[(df['OrderDate'] >= year4) & (df['OrderDate'] < year5)]

    profit = df4['Profit'].sum()
    customers = df4['CustomerID'].nunique()
    region = mode(list(df4['Region']))
    state = mode(list(df4['State']))
    sales = df4['Sales'].sum()

    dataframe4 = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Customers' : [customers],
        'Region' : [region],
        'State' : [state],
        'Year' : [y4],
        'Category' : [title] 
    }
    
    dataframe4 = pd.DataFrame(dataframe4)
   
    dfs_to_combine = (dataframe,dataframe2,dataframe3,dataframe4)

    DFO = pd.concat(dfs_to_combine)
    
    DFO = DFO.set_index(['Category', 'Year'])

    return DFO




furniture_out = getinfo(furniture_df,2014,2015,2016,2017,'Furniture')
tech_out = getinfo(tech_df,2014,2015,2016,2017,'Tech')
office_out = getinfo(office_df,2014,2015,2016,2017,'Office')

out_DFS = (furniture_out,tech_out,office_out)

big_DFO = pd.concat(out_DFS)

#which business segment produces the most sales and profit, and YoY growth for each segment

#function for most sales, profit, customers:

def answers(df,year,data):
    df = pd.DataFrame(df)
    df1 = df[df.index.isin([year],level=1)]
    df_out = df1.loc[df1[data].idxmax()]
    df_out = pd.DataFrame(df_out)
    df_out = df_out.T
    df_out = df_out[data]
    df_out =pd.DataFrame(df_out)
    return df_out


#test
""" x = answers(big_DFO,2017,'Customers')
print(x)  """


#Forcasting

#want to base it on quantity sold per business segment + sales + profit 
##need to get yearly grouped by month for quantity sold + sales + profit



def group(dataframe):
    df1 =dataframe.copy()

    df1['Year'] = dataframe['OrderDate'].dt.to_period('Y')
    df1['Month'] = dataframe['OrderDate'].dt.to_period('M')

    df = df1.groupby(['Year','Month'])[['Sales','Quantity','Profit']].sum()
    df2 = df1.groupby(['Year','Month'])[['CustomerID']].nunique()
    df2 = df2.rename(columns={'CustomerID':'Customers'})
    df3 = pd.merge(df,df2,left_index=True,right_index=True)


    return df3

sales_furniture = group(furniture_df)
""" print(sales_furniture)  """

y= sales_furniture['Sales']
y = y.droplevel(level=0)
y1 = sales_furniture['Profit']
y1 = y1.droplevel(level=0)
y2 = sales_furniture['Quantity']
y2 = y2.droplevel(level=0)


import statsmodels.api as sm
y.index=y.index.to_timestamp()
y1.index=y1.index.to_timestamp()
y2.index=y2.index.to_timestamp()

def seasonal_decompose(y):
    decomposition = sm.tsa.seasonal_decompose(y,model='additive',)
    fig =decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

""" seasonal_decompose(y) """

""" print(y) """


#Building forecasting model based on the SARIMA
#####SALES#####
##testing Stationarity

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

""" test_stationarity(y,'Raw Data') """


#second test of stationarity Augmented Dickery-Fuller Test
from statsmodels.tsa.stattools import adfuller

def ADF_test(timeseries, dataDesc):
    print('> Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - the data is {} stationary with {}% confidence'.format(k,v,'not' if v < dftest[0] else '', 100-int(k[:-1])))

""" ADF_test(y,'Sales') """

""" print(y) """

y_to_train = y[:'2016-12-01']
y_to_val = y[:'2017-12-01']
predict_date = len(y)-len(y_to_val)


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
#print('SARIMA{}x{} - AIC:{}'.format(param,param_seasonal,results.aic))
            except:
                continue
    print('the set of params with min AIC is SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))

""" sarima_grid_search(y,12) """


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


""" model = sarima_eva(y,(0,1,1),(0,1,1,12),12,'2016-12-01',y_to_val) """

##predictive results for sales:

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
#new function for plot
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




""" final_table = forecast(model,12,y)  """


""" print(sales_furniture)  """

###PROFIT###
#1Seasonality Decomposition
#seasonal_decompose(y1)
#2 Stationarity
#test_stationarity(y1,'Profit')
#3 ADF test
#ADF_test(y1,'Profit')
#4Grid search for SARIMA paramaters:
#sarima_grid_search(y1,12)
#5 Model for SARIMA w/PARAMS^
#model_1 = sarima_eva(y1,(0,1,1),(0,1,1,12),12,'2016-12-01',y1_to_val)
#forecast(model,12,y)
y1_to_eval = y1[:'2016-12-01']
y1_to_val = y1[:'2017-12-01']
predict_date = len(y1)-len(y1_to_val)

""" sarima_eva(y1,(0,1,1),(0,1,1,12),12,'2016-12-01',y1_to_eval)

model_1 = sarima_eva(y1,(0,1,1),(0,1,1,12),12,'2016-12-01',y1_to_val)

forecast(model_1,12,y1)  """

#Quantity of Items sold

y2_to_eval = y2[:'2016-12-01']
y2_to_val = y2[:'2017-12-01']
predict_date = len(y2)-len(y2_to_val)


""" ADF_test(y2,'Quantity Sold')
seasonal_decompose(y2)
sarima_grid_search(y2,12)
model_2 = sarima_eva(y2,(0,1,1),(1,1,1,12),12,'2016-12-01',y2_to_eval)
sarima_eva_plots(y2,(0,1,1),(1,1,1,12),12,'2016-12-01',y2_to_eval)
print(model_2)
print(forecast(model_2,12,y2))
forecasted_plot(model_2,12,y2) """

model_2 = sarima_eva(y2,(0,1,1),(1,1,1,12),12,'2016-12-01',y2_to_eval)
####Results:
#SARIMA is a good predictor for sales# and quantity sold: next finding the average profit per unit sold, using the predicted units sold to calculate
#predicted profit amount

F_average_profit_per_unit = round(sales_furniture['Profit'].sum()/sales_furniture['Quantity'].sum(),2)
##Quantity
out_f = forecast(model_2,12,y2)


sales_office = group(office_df)
O_average_profit_per_unit = round(sales_office['Profit'].sum()/sales_office['Quantity'].sum(),2)
sales_tech = group(tech_df)
T_average_profit_per_unit = round(sales_tech['Profit'].sum()/sales_tech['Quantity'].sum(),2)
sales_o = sales_office['Quantity']
sales_o = sales_o.droplevel(level=0)
sales_t = sales_tech['Quantity']
sales_t = sales_t.droplevel(level=0)
sales_o.index=sales_o.index.to_timestamp()
sales_t.index=sales_t.index.to_timestamp()

#ADF_test(sales_o,'Quantity')
#seasonal_decompose(sales_o)
#sarima_grid_search(sales_o,12)
model_o = sarima_eva(sales_o,(0,1,1),(1,1,1,12),12,'2016-12-01',y2_to_eval)
#sarima_eva_plots(sales_o,(0,1,1),(1,1,1,12),12,'2016-12-01',y2_to_eval)
#forecasted_plot(model_o,12,sales_o)

out_o = forecast(model_o,12,sales_o)

#ADF_test(sales_t,'Quantity')
#seasonal_decompose(sales_t)
#sarima_grid_search(sales_t,12)
model_t = sarima_eva(sales_t,(0,1,1),(0,1,1,12),12,'2016-12-01',y1_to_eval)
#sarima_eva_plots(sales_t,(0,1,1),(0,1,1,12),12,'2016-12-01',y1_to_eval)
#forecasted_plot(model_t,12,sales_t)
out_t = forecast(model_t,12,sales_t)


####All three categories are stationary, next calculate average profit per unit sold, using that building a new DF with historical
#profits + units and the forecasted amounts, make another plot with all three plus the lower, median, upper bounded forecasted amounts for the cats



out_o_2 = out_o.copy(deep=True)

O_average_sales_per_unit = round(sales_office['Sales'].sum()/sales_office['Quantity'].sum(),2)
#AVERAGE PROFIT PER UNIT SOLD $5.35

out_o_2[['Predicted_Mean','Lower Bound','Upper Bound']] = out_o_2[['Predicted_Mean','Lower Bound','Upper Bound']] * O_average_profit_per_unit
out_o_2['Date'] = pd.to_datetime(out_o_2['Date'], infer_datetime_format= True)
out_o_2['Year'] = out_o_2['Date'].dt.to_period('Y')
out_o_2['Month'] = out_o_2['Date'].dt.to_period('M')
out_o_2 = out_o_2.drop(columns='Date')
out_o_2 = out_o_2.set_index(['Year', 'Month'])
out_o_2 = out_o_2.rename(columns={'Predicted_Mean' : 'Profit'})
""" print(O_average_sales_per_unit) """# = 31.39
profit_office_array = np.array(out_o_2['Profit'])
quantity_office_array = np.array(out_o['Predicted_Mean'])
sales_office_array = np.array(out_o['Predicted_Mean']*31.39)
index_out = pd.to_datetime(out_o['Date'],infer_datetime_format= True)

#building DF to use to concat with the origin office grouped DF
office_df_to_concat = pd.DataFrame({
    'Date' : index_out,
    'Sales': sales_office_array,
    'Quantity' : quantity_office_array,
    'Profit' : profit_office_array,
})
#cleaning up DF formatting
office_df_to_concat['Year']=office_df_to_concat['Date'].dt.to_period('Y')
office_df_to_concat['Month']=office_df_to_concat['Date'].dt.to_period('M')
office_df_to_concat = office_df_to_concat.drop(columns='Date')
office_df_to_concat = office_df_to_concat.set_index(['Year', 'Month'])


sales_office_out = sales_office.drop(columns='Customers')
#list for the two dfs
dfs = (sales_office_out,office_df_to_concat)

final_office_df = pd.concat(dfs)


##ALL OUT_XXX are for QUANTITY FORECASTS
""" 
print(out_t)
print(out_f)
print(out_o)

"""
###OFFICE SUPPLIES FORECASTS DONE



###FURNITURE
F_average_sales_per_unit = sales_furniture['Sales'].sum()/sales_furniture['Quantity'].sum()
##Furniture average sales per unit sold ========= $92.43
##Furniture average profit per unit sold ========= $2.3
furniture_sales_forecast = np.array(out_f['Predicted_Mean']*F_average_sales_per_unit)
furniture_profit_forecat = np.array(out_f['Predicted_Mean']*F_average_profit_per_unit)
furniture_date = pd.to_datetime(out_f['Date'])
furniture_units = np.array(out_f['Predicted_Mean'])

furniture_df_to_concat = pd.DataFrame(
    {
    'Date' : furniture_date,
    'Sales': furniture_sales_forecast,
    'Quantity' : furniture_units,
    'Profit' : furniture_profit_forecat
}
)

furniture_df_to_concat['Year'] = furniture_df_to_concat['Date'].dt.to_period('Y')
furniture_df_to_concat['Month'] = furniture_df_to_concat['Date'].dt.to_period('M')
furniture_df_to_concat = furniture_df_to_concat.drop(columns='Date')
furniture_df_to_concat = furniture_df_to_concat.set_index(['Year','Month'])
dfs_furniture = (sales_furniture,furniture_df_to_concat)

final_furniture_df = pd.concat(dfs_furniture)
final_furniture_df = final_furniture_df.drop(columns='Customers')


out_f_sales = out_f[['Predicted_Mean','Lower Bound','Upper Bound']] * F_average_sales_per_unit
out_f_profit = out_f[['Predicted_Mean','Lower Bound', 'Upper Bound']] * F_average_profit_per_unit
out_o_sales = out_o[['Predicted_Mean','Lower Bound', 'Upper Bound']]* O_average_sales_per_unit
out_o_profit = out_o[['Predicted_Mean','Lower Bound', 'Upper Bound']] * O_average_profit_per_unit

##Tech forecast + Historicals

T_average_sales_per_unit = sales_tech['Sales'].sum()/sales_tech['Quantity'].sum()

out_t_sales = out_t[['Predicted_Mean', 'Lower Bound', 'Upper Bound']] * T_average_sales_per_unit
out_t_profit = out_t[['Predicted_Mean', 'Lower Bound', 'Upper Bound']] * T_average_profit_per_unit

sales_tech_forecast = np.array(out_t_sales['Predicted_Mean'])
profit_tech_forecast = np.array(out_t_profit['Predicted_Mean'])
tech_date = pd.to_datetime(out_t['Date'])
tech_units = np.array(out_t['Predicted_Mean'])

tech_df_to_concat = pd.DataFrame(
    {
    'Date' : tech_date,
    'Sales': sales_tech_forecast,
    'Quantity' : tech_units,
    'Profit' : profit_tech_forecast
}
)

tech_df_to_concat['Year'] = tech_df_to_concat['Date'].dt.to_period('Y')
tech_df_to_concat['Month'] = tech_df_to_concat['Date'].dt.to_period('M')
tech_df_to_concat = tech_df_to_concat.drop(columns='Date')
tech_df_to_concat = tech_df_to_concat.set_index(['Year','Month'])
dfs_tech = (sales_tech,tech_df_to_concat)

final_tech_df = pd.concat(dfs_tech)
final_tech_df = final_tech_df.drop(columns='Customers')



##from here plots

##sales

sales_furniture_plot = np.array(final_furniture_df['Sales'])
sales_tech_plot = np.array(final_tech_df['Sales'])
sales_office_plot = np.array(final_office_df['Sales'])
profit_furniture_plot = np.array(final_furniture_df['Profit'])
profit_office_plot =np.array(final_office_df['Profit'])
profit_tech_plot = np.array(final_tech_df['Profit'])

#getting x-value-timeseries
new_axis_df = final_furniture_df.droplevel(level=0)

new_axis_df = new_axis_df.reset_index()

new_axis_df['Month'] = new_axis_df['Month'].dt.to_timestamp('s').dt.strftime('%Y-%m')

x_axis = new_axis_df['Month']

x_TICKS = 12
plt.figure(figsize=(18,7))
plt.plot(x_axis, sales_furniture_plot, 'r', label = 'Furniture Sales')
plt.plot(x_axis,sales_tech_plot,'b', label = 'Tech Sales')
plt.plot(x_axis, sales_office_plot,'c', label = 'Office Sales')
plt.xticks(range(0, len(x_axis), x_TICKS),x_axis[::x_TICKS], rotation = 45)
plt.title('Sales Historical + Forecasted \n Furniture Office Tech')
plt.legend()
plt.show()

x_TICKS = 12
plt.figure(figsize=(18,7))
plt.plot(x_axis,profit_furniture_plot,'r', label = 'Furniture Profit')
plt.plot(x_axis,profit_office_plot,'b', label = 'Office Profit')
plt.plot(x_axis,profit_tech_plot,'c', label = 'Tech Profit')
plt.xticks(range(0, len(x_axis), x_TICKS),x_axis[::x_TICKS], rotation = 45)
plt.title('Profit Historical + Forecasted \n Furniture Office Tech')
plt.legend()
plt.show()


#OutPut_DF.to_csv(r"C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\test.csv",index=False)

















