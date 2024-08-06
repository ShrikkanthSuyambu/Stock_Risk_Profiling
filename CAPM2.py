#!/usr/bin/env python
# coding: utf-8

# In[2]:


from copy import copy
import numpy as np
import pandas as pd


# In[3]:


df = pd.read_excel(r"C:\Users\Shrikkanth Suyambu\OneDrive\Desktop\CAPM Finance Project\Dataset\CAPM Data Set.xlsx")


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


#creating a function to calculate daily returns
def daily_return(df):
    #using the copy function to avoid changes to orginal data
    df_daily_return = df.copy()
    #looping through every column
    for i in df.columns[1:]:
        #looping through every row
        for j in range(1, len(df)):
            #calculating daily returns
            df_daily_return.loc[j, i] = ((df.loc[j,i]-df.loc[j-1,i]) / df.loc[j-1,i])*100
            #setting the return of first row to zero
            df_daily_return.loc[0,i] = 0
    return df_daily_return


# In[7]:


# getting the daily returns of the stock

stock_daily_return = daily_return(df)
stock_daily_return.head()


# In[8]:


#calculating beta of TATASTEEL

TATASTEEL_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['TATASTEEL'],1)
print('Beta for {} stock is = {}'.format('TATASTEEL', TATASTEEL_Beta))


# In[9]:


#calculating beta of NTPC

NTPC_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['NTPC'],1)
print('Beta for {} stock is = {}'.format('NTPC', NTPC_Beta))


# In[10]:


#calculating beta of ONGC

ONGC_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['ONGC'],1)
print('Beta for {} stock is = {}'.format('ONGC', ONGC_Beta))


# In[11]:


#calculating beta of TATAMOTORS

TATAMOTORS_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['TATAMOTORS'],1)
print('Beta for {} stock is = {}'.format('TATAMOTORS', TATAMOTORS_Beta))


# In[12]:


#calculating beta of ITC

ITC_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['ITC'],1)
print('Beta for {} stock is = {}'.format('ITC', ITC_Beta))


# In[13]:


#calculating beta of BPCL

BPCL_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['BPCL'],1)
print('Beta for {} stock is = {}'.format('BPCL', BPCL_Beta))


# In[14]:


#calculating beta of POWERGRID

POWERGRID_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['POWERGRID'],1)
print('Beta for {} stock is = {}'.format('POWERGRID', POWERGRID_Beta))


# In[15]:


#calculating beta of HDFCBANK

HDFCBANK_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['HDFCBANK'],1)
print('Beta for {} stock is = {}'.format('HDFCBANK', HDFCBANK_Beta))


# In[16]:


#calculating beta of SBIN

SBIN_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['SBIN'],1)
print('Beta for {} stock is = {}'.format('SBIN', SBIN_Beta))


# In[17]:


#calculating beta of COALINDIA

COALINDIA_Beta, _ = np.polyfit(stock_daily_return['NIFTY500'], stock_daily_return['COALINDIA'],1)
print('Beta for {} stock is = {}'.format('COALINDIA', COALINDIA_Beta))


# In[18]:


#Calculating market return
rm = stock_daily_return['NIFTY500'].mean()*248


# In[19]:


#calculating risk free return

rf = 7.10/100


# In[20]:


# calculating Expected return using CAPM
# RESULTS ARE IN PERCENTAGES


# In[21]:


#TATASTEEL Expected return using CAPM Model

ER_TATASTEEL = rf + (TATASTEEL_Beta * (rm-rf))
print(ER_TATASTEEL)


# In[22]:


#NTPC Expected return using CAPM Model

ER_NTPC = rf + (NTPC_Beta * (rm-rf))
print(ER_NTPC)


# In[23]:


#ONGC Expected return using CAPM Model

ER_ONGC = rf + (ONGC_Beta * (rm-rf))
print(ER_ONGC)


# In[24]:


#TATAMOTORS Expected return using CAPM Model

ER_TATAMOTORS = rf + (TATAMOTORS_Beta * (rm-rf))
print(ER_TATAMOTORS)


# In[25]:


#ITC Expected return using CAPM Model

ER_ITC = rf + (ITC_Beta * (rm-rf))
print(ER_ITC)


# In[26]:


#BPCL Expected return using CAPM Model

ER_BPCL = rf + (BPCL_Beta * (rm-rf))
print(ER_BPCL)


# In[27]:


#POWERGRID Expected return using CAPM Model

ER_POWERGRID = rf + (POWERGRID_Beta * (rm-rf))
print(ER_POWERGRID)


# In[28]:


#HDFCBANK Expected return using CAPM Model

ER_HDFCBANK = rf + (HDFCBANK_Beta * (rm-rf))
print(ER_HDFCBANK)


# In[29]:


#SBIN Expected return using CAPM Model

ER_SBIN = rf + (SBIN_Beta * (rm-rf))
print(ER_SBIN)


# In[30]:


#COALINDIA Expected return using CAPM Model

ER_COALINDIA = rf + (COALINDIA_Beta * (rm-rf))
print(ER_COALINDIA)


# In[39]:


#stock and their expected return
data = {
    'Stock': ['TATASTEEL', 'NTPC', 'ONGC', 'TATAMOTORS', 'ITC', 'BPCL', 'POWERGRID', 'HDFCBANK', 'SBIN', 'COALINDIA'],
    'expected_return': [ER_TATASTEEL, ER_NTPC, ER_ONGC, ER_TATAMOTORS, ER_ITC, ER_BPCL, ER_POWERGRID, ER_HDFCBANK, ER_SBIN, ER_COALINDIA],
    'Stock_Beta': [TATASTEEL_Beta, NTPC_Beta, ONGC_Beta, TATAMOTORS_Beta, ITC_Beta, BPCL_Beta, POWERGRID_Beta, HDFCBANK_Beta, SBIN_Beta, COALINDIA_Beta]
}
df = pd.DataFrame(data)


# In[1]:


#K-MEANS clustering

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



# Function to classify stock risk using K-Means clustering
@st.cache_data
def classify_risk(data):

    kmeans = KMeans(n_clusters=3, random_state=0)
    data['risk_category'] = kmeans.fit_predict(data[['expected_return','Stock_Beta']])
   
    return data, kmeans.cluster_centers_

# Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: white; background-color: #2C3E50; padding: 10px;'>Portfolio Management</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #5DADE2; padding: 10px;'>
        <h3 style='color: white;'>Asset: Stocks</h3>
        <h4 style='color: white;'>Stock Exchange: National Stock Exchange</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='background-color: #D6EAF8; padding: 10px;'><h4>Choose your risk profile</h4></div>", unsafe_allow_html=True)


    # Classify risk
    classified_data, cluster_centers = classify_risk(df)

    # Map risk categories to human-readable labels
    risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    classified_data['risk_category'] = classified_data['risk_category'].map(risk_labels)

    # Dropdown to select risk profile
    risk_profile = st.selectbox('Select a Risk Profile:', ['Low Risk', 'Medium Risk', 'High Risk'])

    # Filter the data based on selected risk profile
    filtered_data = classified_data[classified_data['risk_category'] == risk_profile]
    

    # Add numbering to the stocks
    filtered_data.reset_index(drop=True, inplace=True)
    filtered_data.index += 1

    # Display classified data in a table with enhanced styling
    st.markdown(f"<h2>Stocks under {risk_profile}</h2>", unsafe_allow_html=True)
    st.markdown(filtered_data[['Stock', 'expected_return']].rename(columns={'Stock': 'Stock Name', 'expected_return': 'Expected Return'}).to_html(index=True, bold_rows=True), unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# In[ ]:




