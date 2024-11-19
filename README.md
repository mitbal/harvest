### Preamble
Panen Dividen (https://harvest-dividend.up.railway.app/) is a sophisticated analytics tool designed to optimize your dividend investment strategy. It offers comprehensive insights into past performance metrics and provides predictive analysis for future yield potential.

If you find this app to be remotely useful and help you on your financial independent journey, please consider donation to support the development.

<img src="https://raw.githubusercontent.com/mitbal/harvest/refs/heads/master/asset/trakteer_icon.png" width="13%"> [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/mitbal)

You can report any bug or post feature request on the github issue
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/mitbal/harvest)](https://github.com/mitbal/harvest/issues) 

### How to use?
This app currently has 3 modules.

#### 1. Stock Screener
The Stock Screener module assists you in selecting stocks to include in your portfolio. It lists all available stocks on the Jakarta Stock Exchange, filtering and ranking them based on their potential for future dividend payments.
![sp 1](https://raw.githubusercontent.com/mitbal/harvest/refs/heads/master/asset/sp1.png)
You can also view it in the form of 2D scatter plot, where you can select different attributes for each axes.
![sp 2](https://raw.githubusercontent.com/mitbal/harvest/refs/heads/master/asset/sp2.png)
By selecting a single stock, you can view its historical dividend payment, financial information and its daily price for the last 1 year.
![sp 3](https://raw.githubusercontent.com/mitbal/harvest/refs/heads/master/asset/sp3.png)
![sp 4](https://raw.githubusercontent.com/mitbal/harvest/refs/heads/master/asset/sp4.png)


#### 2. Portfolio Analysis 
The Portfolio Analysis module offers advanced predictive analytics to estimate your future income from dividends based on your current portfolio. 

There are multiple methods to input your data:
1. **Manual Entry**: Use the provided form to input your portfolio data one stock at a time.
2. **CSV Upload**: If you have your portfolio data in a CSV file, you can upload it directly. Ensure your CSV follows this schema, with each column separated by a comma.

By utilizing these methods, you can seamlessly analyze your portfolio's potential for future dividend income.
```data
Symbol,Available Lot,Average Price
ACES,68,860.62
ADRO,121,"3,181.48"
```
The data needed can be taken from your favorite security, like Stockbit, Ajaib, or Indopremier.


#### 3. Historical Insight
The first module perform a descriptive analysis on historical dividend payment transactions.
All you need to do is provide a csv file following this sample schema with semicolon as delimiter:
```data
Date;Stock;Lot;Price;Total Dividend
2024-07-12;IPCC;102;62.39;636378
2024-07-12;BIRD;101;91;919100
```
