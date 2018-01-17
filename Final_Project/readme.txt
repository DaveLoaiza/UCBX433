Python For Data Analysis And Scientific Computing
UC Berkeley Extension

Final Project -- Stock Performance Calculator

David Loaiza -- dave.loaiza@gmail.com
John Graham -- johngraham415@gmail.com

The purpose of this project is to create a basic stock performance calculator. The presentation includes functions to input stock purchases at
the lot level. It stores, in CSV text file format, a stocks symbol(or ticker), the purchase quantity, the purchase price, and the purchase date.
Each time a stock is purchased, the file is appended to continue building a portfolio. Additionally, it includes functions to load the porfolio
file data into a pandas numpy array, and calculate performance and unrealized gain loss.

The code was created in Python v3.6

The following modules/packages are required, and can be installed using pip:
numpy
scipy
pandas
pandas-datareader
yahoo-finance
pylab
statsmodels