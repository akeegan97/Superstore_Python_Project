
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

#Getting the different main categories of goods sold
categories = raw_data.Category.unique()

#rename the column titles to remove - 



raw_data.rename(columns={'Sub-Category' : 'SubCategory'}, inplace = True)
raw_data.rename(columns={'Order ID' : 'OrderID'}, inplace= True)
raw_data.rename(columns={'Product Name' : 'ProductName'}, inplace= True)
raw_data.rename(columns={'Order Date' : 'OrderDate'}, inplace= True)
raw_data.rename(columns={'Customer ID' : 'CustomerID'}, inplace= True)
####Investigate making this a DataFrame wide Function that removes spaces and special characters, to aid in calling specific columns^^^^
raw_data['OrderDate'] = pd.to_datetime(raw_data['OrderDate'], infer_datetime_format= True)

subcategories = raw_data.SubCategory.unique()

#just getting Main Category data frames 
""" print("main categories: \n", categories)  
print('Column Headers are :\n', raw_data.columns) """
furniture_df = raw_data.loc[raw_data['Category'] == 'Furniture']
office_df = raw_data.loc[raw_data['Category'] == 'Office Supplies']
tech_df = raw_data.loc[raw_data['Category']=='Technology'] 


#subcategories in each group
furniture_subcat = furniture_df.SubCategory.unique()
office_subcat = office_df.SubCategory.unique()
tech_subcat = tech_df.SubCategory.unique()

#finding the different profits for the three main categories


#count of orders for each main categories

cnt_furniture = furniture_df.OrderID.nunique()
cnt_office = office_df.OrderID.nunique()
cnt_tech = tech_df.OrderID.nunique()

print('columns in subcategory df :\n', furniture_df.columns) 

quantity_furniture = furniture_df.Quantity.sum()
quantity_office = office_df.Quantity.sum()
quantity_tech = tech_df.Quantity.sum()

""" print('Unique Orders for Furniture :\n', cnt_furniture)
print('Unique Orders for Office :\n', cnt_office)
print('Unique Orders for Tech :\n', cnt_tech)

print('total number of goods sold for furniture: \n', quantity_furniture)
print('total number of goods sold for Office: \n', quantity_office)
print('total number of goods sold for Technology: \n', quantity_tech)  """

total_number_of_goods_sold = quantity_tech + quantity_furniture + quantity_office

""" print('total number of goods sold :\n', total_number_of_goods_sold) """
#datetime in format YYYY-MM-DD


#breaking the Raw_data into 3 main category dfs with the useful column information

def filtered(dataframe):
    newdf = dataframe[['OrderDate', 'OrderID','Category','CustomerID','SubCategory','ProductName','Quantity','Sales','Discount','Profit']]
    return newdf

office_filtered = filtered(office_df)
tech_filtered = filtered(tech_df)
furniture_filtered = filtered(furniture_df)


#breaking those further down into yearly outputs

def yearly(dataframe, start, end):
    start_date = pd.Timestamp(start, 1, 1)
    end_date = pd.Timestamp(end, 12, 31)
    newdf = dataframe[(dataframe['OrderDate'] >= start_date) & (dataframe['OrderDate'] <= end_date)]
    return newdf

furniture_2014 = yearly(furniture_filtered,2014,2014)
office_2014 = yearly(office_filtered,2014,2014)
tech_2014 = yearly(tech_filtered, 2014,2014)

#grouping sales,profit,quantity by months for a length 12 and width 4 df 

def group(dataframe):
    df = dataframe.groupby(dataframe.OrderDate.dt.month)[['Profit', 'Sales','Quantity']].sum()

    df['Months'] = 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
    return df


#getting df with useful statistics of the dfs

def stats(dataframe,start,end):
    st = pd.Timestamp(start, 1, 1)
    ed = pd.Timestamp(end, 12, 31)

    df2 = dataframe[(dataframe['OrderDate'] >= st) & (dataframe['OrderDate'] <= ed)]

    sales = df2['Sales'].sum()
    profit = df2['Profit'].sum()
    units_sold = df2['Quantity'].sum()
    unique_customers = df2['CustomerID'].nunique()
    unique_orders = df2['OrderID'].nunique()
    orders = df2['OrderID'].count()
    average_quantity_p_order = df2['Quantity'].sum() // orders
    profit_p_order = profit // unique_orders
    profit_p_customer = profit // unique_customers

    df = {
        'Sales' : [sales],
        'Profit' : [profit],
        'Units_Sold' : [units_sold],
        'Customers' : [unique_customers],
        'Orders' : [orders],
        'PPO' : [profit_p_order],
        'PPC' : [profit_p_customer],
        'APO' : [average_quantity_p_order]
    }
    df = pd.DataFrame(df)
    return df 

### plotting the monthly data for the grouped dfs furniture, office, tech


grouped_tech_2014 = group(tech_2014)
""" grouped_office_2014 = group(office_2014)
grouped_furniture_2014 = group(furniture_2014) """

#tech2014-2017
tech_2015 = yearly(tech_filtered,2015,2015)
grouped_tech_2015 = group(tech_2015)
tech_2016 = yearly(tech_filtered,2016,2016)
tech_2017 = yearly(tech_filtered,2017,2017)


grouped_tech_2016 = group(tech_2016)
grouped_tech_2017 = group(tech_2017)
dfs_tech = (grouped_tech_2014,grouped_tech_2015,grouped_tech_2016,grouped_tech_2017)

tech_2014_2017 = pd.concat(dfs_tech, ignore_index=True)

#furniture2014-2017

furniture_2015 = yearly(furniture_filtered,2015,2015)
furniture_2016 = yearly(furniture_filtered,2016,2016)
furniture_2017 = yearly(furniture_filtered,2017,2017)
grouped_furniture_2014 = group(furniture_2014)
grouped_furniture_2015 = group(furniture_2015)
grouped_furniture_2016 = group(furniture_2016)
grouped_furniture_2017 = group(furniture_2017)

dfs_furn = (grouped_furniture_2014,grouped_furniture_2015,grouped_furniture_2016,grouped_furniture_2017)
furn_2014_2017 = pd.concat(dfs_furn, ignore_index=True)

#Office2014-2017

office_2015 = yearly(office_filtered,2015,2015)
office_2016 = yearly(office_filtered,2016,2016)
office_2017 = yearly(office_filtered,2017,2017)
g_off_2014 = group(office_2014)
g_off_2015 = group(office_2015)
g_off_2016 = group(office_2016)
g_off_2017 = group(office_2017)

dfs_off = (g_off_2014,g_off_2015,g_off_2016,g_off_2017)
of_2014_2017 = pd.concat(dfs_off, ignore_index=True)

office_profit_line = of_2014_2017['Profit']
furniture_profit_line = furn_2014_2017['Profit']
tech_profit_lines = tech_2014_2017['Profit']

""" 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
major_tick = [0,11,23,35,47]
minor_tick = [5,17,29,41]
ax.set_xticks(major_tick, minor = False)
ax.set_xticks(minor_tick, minor = True)
plt.plot(office_profit_line, label = 'Office Supplies')
plt.plot(furniture_profit_line, label = "Furniture")
plt.plot(tech_profit_lines, label = 'Technology')
plt.legend()
plt.grid(True)
plt.show() """


#most profit generating business segment:

furniture_profit = furniture_filtered['Profit'].sum()
office_profit = office_filtered['Profit'].sum()
tech_profit = tech_filtered['Profit'].sum()

output_df = {
    'Furniture' : [furniture_profit],
    'Office': [office_profit],
    'Tech' : [tech_profit]
}

output_df = pd.DataFrame(output_df)
output_df = output_df.round(2)

#unique customers per segment:

f_customers = furniture_filtered['CustomerID'].nunique()
o_customers = office_filtered['CustomerID'].nunique()
t_customers = tech_filtered['CustomerID'].nunique()

customer_df = {
    'Furniture' : [f_customers],
    'Office' :[o_customers],
    'Tech' :[t_customers]
}

customer_df = pd.DataFrame(customer_df)

print(customer_df)
#most common geographic area:
f_region = mode(list(furniture_df['Region']))
o_region = mode(list(office_df['Region']))
t_region = mode(list(tech_df['Region']))

region_df = {
    'Furniture' : [f_region],
    'Office' : [o_region],
    'Tech' :[t_region]
}
region_df = pd.DataFrame(region_df)






#most common State:

f_state = mode(list(furniture_df['State']))
o_state = mode(list(office_df['State']))
t_state = mode(list(tech_df['State']))


""" print(f_state) """
state_df = {
    'Furniture' : [f_state],
    "Office" : [o_state],
    'Tech' : [t_state]
}

state_df = pd.DataFrame(state_df)




facts = (output_df,customer_df,region_df,state_df)
OutPut_DF = pd.concat(facts,ignore_index=True)
indexdf = ['Profit', 'Customers', 'Region', 'State']
""" print(OutPut_DF) """

OutPut_DF.insert(0,'Categories', indexdf)
OutPut_DF.set_index(['Categories'])


print(OutPut_DF) 

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



#OutPut_DF.to_csv(r"C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\test.csv",index=False)




