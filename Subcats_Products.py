
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
###Drill down on the subcategories and products sold for each main category

super_store_data  = r'C:\Users\Andrew\Desktop\Python_Projects\Super_Market_Project\Sample - Superstore.csv'
main_data = pd.read_csv(super_store_data,encoding='Latin')
main_data.columns = main_data.columns.str.replace(' ','')
main_data.columns = main_data.columns.str.replace('-','')


products = main_data['SubCategory'].nunique()


column_names = main_data.columns



furniture = main_data.loc[main_data['Category'] == 'Furniture']
tech = main_data.loc[main_data['Category'] == 'Technology']
office = main_data.loc[main_data['Category'] == 'Office Supplies']



""" furniture.SubCategory.hist()

tech.SubCategory.hist()

office.SubCategory.hist() """

###show the trends of the subcategories of each main category based on orders and one on profit amount


Furniture_Unique_Cats = ['Bookcases','Chairs', 'Tables', 'Furnishings']
Office_Unique_Cats = ['Labels' ,'Storage' ,'Art', 'Binders', 'Appliances' ,'Paper', 'Envelopes', 'Fasteners', 'Supplies']
Tech_Unique_Cats = ['Phones', 'Accessories', 'Machines', 'Copiers']












