#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:21:37 2021

@author: ekdev
"""

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

web_url = "https://finviz.com/quote.ashx?t=" # finviz url

tickers_list = []
while True:
    user_input = input('Enter Company: ')
    answer = input('More Companies? Y/N: ')
    
    if answer == 'Y' or answer == 'y':
        tickers_list.append(user_input)
        continue
    elif answer == 'N' or answer == 'n':
        tickers_list.append(user_input)
        break

#tickers_list = ['AMZN', 'GOOG', 'FB'] # tickers to parse

news_data = {}

for ticker in tickers_list: # appending names of tickers onto url
    url = web_url + ticker
    
    req = Request(url = url, headers = {'user-agent': 'my-app'}) # request html data
    response = urlopen(req) # object to parse data out of 
    
    html = BeautifulSoup(response, 'html') # view html source code
    news_table = html.find(id = 'news-table') # get html object of news table in finviz
    news_data[ticker] = news_table # store table object into a dictionary
    

parsed_data = []

for ticker, news_data in news_data.items():
    
    for row in news_data.findAll('tr'):
        
        title = row.a.text
        date_data = row.td.text.split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns =['ticker', 'date', 'time', 'title']) # pass in array to pandas data frame

vader = SentimentIntensityAnalyzer() # analyze given text

funct = lambda title: vader.polarity_scores(title)['compound'] # function to get the sentiment compound score
df['compound'] = df['title'].apply(funct) # apply function to each title
df['date'] = pd.to_datetime(df.date).dt.date # convert date column to datetime format

plt.figure(figsize = (10,8))

mean_df = df.groupby(['ticker', 'date']).mean() # date frame displaying average sentiment for each date
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis = "columns").transpose() # taking cross-section to remove compound column

mean_df.plot(kind = 'bar') # plot bar graph
plt.show()





