# -*- coding: utf-8 -*-
"""
Python For Data Analysis And Scientific Computing
UC Berkeley Extension

Final Project -- Stock Performance Calculator

David Loaiza -- dave.loaiza@gmail.com
John Graham -- johngraham415@gmail.com

#    """
   
from datetime import date
import numpy as np
from os import path
import csv
import simplejson as json
import yahoo_finance as yf
import pandas
import pandas_datareader.data as web
from pylab import *
from scipy import *
from statsmodels.tsa.arima_model import ARIMA

# function for writing stock purchases to file, ie. builds portfolio record at lot level(each stock purchase instance)
def update_portfolio(TICKER, QUANT, PRICE, DAY=date.today().day, MON=date.today().month, YEAR=date.today().year):
    """Writes stock purchases to open_lots.txt file which stores portfolio
    TICKER format is str4
    QUANT format is int
    PRICE format is float
    DAY format is int, defaults to current day
    MON format is int, defaults to current month
    YEAR format is int, defautls to curent year
    open_lots.txt is a comma separated file, with each line representing a stock purchase lot: TICKER,QUANTITY,PRICE,PURCHASEDATE
    """ 
    
    if path.exists('final_project/open_lots.txt') == True:      #Check to see if file exists
        open_lot = open('final_project/open_lots.txt', 'a')     #if exists, open in 'Append' mode
    else:
        open_lot = open('final_project/open_lots.txt', 'w')     #else, opens in 'write' mode to create first time
    
    DATE = date(YEAR, MON, DAY)
    open_lot.writelines(str(TICKER)+','+str(QUANT)+','+str(PRICE)+','+str(DATE)+'\n')
    open_lot.close()
    return 'Adding purchase of %s shares of %s at a price of $ %s on %s to portfolio.' % (QUANT, TICKER, PRICE, DATE)

# function to read the open_lots.txt file and return a list of unique stock tickers in portfolio
def stock_list():
    """ Parses the open_lots.txt portfolio file and returns a list of the stock tickers of those stocks held in the Portfolio """
    open_lot = open('final_project/open_lots.txt', 'r')     #open open_lots.txt file for reading
    stock_list = []
    reader = csv.DictReader(open_lot, fieldnames=('Stock', 'Quant', 'Price', 'Date'))
    for row in reader:
        stock_list.append(row['Stock'])
    stock_list = list(set(stock_list))
    return stock_list                                       #returns a list that can be iterated through to find daily prices

# function to pull stock prices for a list of stocks:
def get_price(stock_list):
    """ Takes a list of strings (stock tickers), and uses each ticker string to query Yahoo finance API and pull most current share price """
    stock_prices = {'Share':'Latest Price'}                 #dictionary for storing stock/price pairs
    curr_price = open('final_project/curr_price.txt', 'w')  #file for writing latest stock price
    for s in stock_list:                                    #loop through list of stocks, query Yahoo with each
        share = yf.Share(s)
        c_price = share.get_price()
        stock_prices.update({s:c_price})
#    curr_price.write(str(stock_prices))
    json.dump(stock_prices,curr_price)    
    curr_price.close()
    return stock_prices

# function to pull historical stock prices

"""  HISTORICAL DATA MODULE NOT WORKING
def get_price_hist():
    #  Function parses open_lots.txt portfolio file, returns historical data including daily prices for all stocks since their purchase date 
    open_lot = open('final_project/open_lots.txt', 'r')       #file containing all open lots
    reader = csv.DictReader(open_lot, fieldnames=('Stock', 'Quant', 'Price', 'Date'))
    stock_dates = {}                                          #to be used to populate Yahoo query with stock tickers and dates for historical data 
    for row in reader:                                        #populate stock_dates dictionary
        stock_dates.update({row['Stock']:row['Date']})
    print(stock_dates)
    open_lot.close()
    price_hist = open('final_project/price_hist.txt', 'w')    #file for writing historical price data
    today = str(date.today().year)
    if len(str(date.today().month)) == 1:
        today = today+'-0'+str(date.today().month)
    else:
        today = today+'-'+str(date.today().month)
    if len(str(date.today().day)) == 1:
        today = today+'-0'+str(date.today().day)
    else:
        today = today+'-'+str(date.today().day)
    print(today)
    for stock, date1 in stock_dates.items():                            #iterate through stock_dates dictionary
            share = yf.Share(stock)
            print(stock + date1 + today)
            print(share.get_historical(date1, today))                                
            json.dump(share.get_historical(date1, today),price_hist)
            price_hist.close()
            price_hist = open('final_project/price_hist.txt', 'a')
    price_hist.close()
"""    
 
# import open lots csv into pandas array
def importy(filename):
    """ this function imports the portfolio as created in update_portfolio function into a pandas numpy array """
    df = 7
    df = pandas.read_csv(filename)
    df.columns = ['Stock', 'Quant', 'Price', 'Date']
    return df

# create basic function for getting stock price for an individual stock
def getprice(stockname):
    share = yf.Share(stockname)
    return share.get_price()

#load portfolio to a pandas numpy array and update with calculated values of performance
def portfolio_calc(filename):
    importy(filename)
    df['CurrentPrice'] = df['Stock'].apple(getprice)
    df['CurrentPrice'] = df['CurrentPrice'].astype(float)
    df['Performance'] = (df['CurrentPrice'] - df['Price'])/df['Price']*100
    df['Unrealized Gain/Loss'] = df['Quant']*(df['CurrentPrice']-df['Price'])
    df.head()
    return df


    