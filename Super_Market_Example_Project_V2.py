
#Given a set dataset of key variables from a fictional superstore.
# file path of data : "C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\Sample - Superstore.csv"
#Dataset given by Kaggle.com open datasets

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
""" print(sales_furniture) """

y= sales_furniture['Sales']
y = y.droplevel(level=0)
y1 = sales_furniture['Profit']

import statsmodels.api as sm
y.index=y.index.to_timestamp()
def seasonal_decompose(y):
    decomposition = sm.tsa.seasonal_decompose(y,model='additive',)
    fig =decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

seasonal_decompose(y)

""" print(y) """





















#OutPut_DF.to_csv(r"C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\test.csv",index=False)

















