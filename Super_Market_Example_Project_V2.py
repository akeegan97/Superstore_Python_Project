
#Given a set dataset of key variables from a fictional superstore. 
#Want to find some insight from the dataset with regards to the most valuable customer class, most profitable product, region, etc

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

#rename the column titles to remove - 



raw_data.rename(columns={'Sub-Category' : 'SubCategory'}, inplace = True)
raw_data.rename(columns={'Order ID' : 'OrderID'}, inplace= True)
raw_data.rename(columns={'Product Name' : 'ProductName'}, inplace= True)
raw_data.rename(columns={'Order Date' : 'OrderDate'}, inplace= True)
raw_data.rename(columns={'Customer ID' : 'CustomerID'}, inplace= True)

####Investigate making this a DataFrame wide Function that removes spaces and special characters, to aid in calling specific columns^^^^
#Changing standard date to PD DATETIME

raw_data['OrderDate'] = pd.to_datetime(raw_data['OrderDate'], infer_datetime_format= True)


#breaking DF into the three main categories


furniture_df = raw_data.loc[raw_data['Category'] == 'Furniture']
office_df = raw_data.loc[raw_data['Category'] == 'Office Supplies']
tech_df = raw_data.loc[raw_data['Category']=='Technology'] 


""" #subcategories in each group
furniture_subcat = furniture_df.SubCategory.unique()
office_subcat = office_df.SubCategory.unique()
tech_subcat = tech_df.SubCategory.unique() """

#datetime in format YYYY-MM-DD



#Function to filter Main DFS by year

def yearly(dataframe, start, end):
    start_date = pd.Timestamp(start, 1, 1)
    end_date = pd.Timestamp(end, 12, 31)
    newdf = dataframe[(dataframe['OrderDate'] >= start_date) & (dataframe['OrderDate'] <= end_date)]
    return newdf


#grouping sales,profit,quantity by months for a length 12 and width 3 df 

def group(dataframe):
    df = dataframe.groupby(dataframe.OrderDate.dt.month)[['Profit', 'Sales','Quantity']].sum()

    df['Months'] = 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
    return df

#most profit generating business segment:


#need function to take raw_data filter yearly, filter category, filter [profit,state,region,customerID] return df with ex.
#furniture:
#profit - xx
#customer - yy
#state - zz
#region - aa

furniture = 'Furniture'
office = 'Office'
tech = 'Tech'




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




furniture_out = getinfo(furniture_df,2014,2015,2016,2017,furniture)
tech_out = getinfo(tech_df,2014,2015,2016,2017,tech)
office_out = getinfo(office_df,2014,2015,2016,2017,office)

out_DFS = (furniture_out,tech_out,office_out)

big_DFO = pd.concat(out_DFS)
print(big_DFO)

#which business segment produces the most sales and profit, and YoY growth for each segment


#Profit2014

big_14 = big_DFO[big_DFO.index.isin(['2014'], level=1)]
print(big_14)

profit_2014 = big_14.loc[big_14['Profit'].idxmax()]
print(profit_2014)
profit_2014 = pd.DataFrame(profit_2014)
profit_2014 = profit_2014.T
print(profit_2014)



















#OutPut_DF.to_csv(r"C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\test.csv",index=False)

















