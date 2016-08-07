import pandas as pd
import datetime as dt
import util as ut
import csv
import StrategyLearner as sl
import numpy as np
import matplotlib.pyplot as plt
import time

def test_code(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "ML4T-220"#
    sym = "IBM"#

    stdate =dt.datetime(2007,12,31)
    enddate =dt.datetime(2009,12,31)
    
    deb = time.time()
    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate,ed = enddate, sv = 10000) 
    
    stdate =dt.datetime(2007,12,31)
    enddate =dt.datetime(2009,12,31)
#     stdate =dt.datetime(2007,12,31)
#     enddate =dt.datetime(2008,1,10)
#     stdate =dt.datetime(2009,12,31)
#     enddate =dt.datetime(2011,12,31)



    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices


    df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 10000)
    print time.time()-deb
    
    
    deliverableB = open("bench.csv", 'w')   
    deliverableB.write("Date,Symbol,Order,Shares\n")
    writeLine(deliverableB,sym,stdate,"BUY",100)
    writeLine(deliverableB,sym,enddate,"SELL",100)
    deliverableB.close()
    
    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 100, 0, -100
    if isinstance(df_trades, pd.DataFrame) == False:
		print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
		print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=100] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
		print "Returned result violoates holding restrictions (more than 100 shares)"

    if verb: print df_trades
    
    plt.figure(5) #Will plot return along with SPX
    marketSimulation("orderStrat.csv")
   # plt.figure(6)
   # marketSimulation("bench.csv")
    plt.show()
    # we will add code here to evaluate your trades

def marketSimulation(filename):

    rfr = 0 
    sf = 252

    # Process orders
    portvals = compute_portvals(orders_file = filename, start_val = 10000)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio  = assess_portfolio(portvals)
    
    dates = pd.date_range(start_date, end_date)
    prices_SPY = ut.get_data(["$SPX"], dates)["$SPX"] # automatically adds SPY
    #prices_SPY = prices_SPY/prices_SPY.ix[0] 
    cum_ret_SPY = (prices_SPY[-1]-prices_SPY[0]) / prices_SPY[0]  #last -first /first
    
    daily_return_SPY = (prices_SPY[1:].values/prices_SPY[:-1].values) - 1  #difference between end of day and beginning of day

    
    plt.plot(portvals/portvals[0],'r',label='Portfolio')
    plt.plot(prices_SPY/prices_SPY[0],'g',label='$SPX')
    plt.ylabel('Normalized price')
    plt.legend()
    


    avg_daily_ret_SPY = daily_return_SPY.mean()
    std_daily_ret_SPY = daily_return_SPY.std(ddof=1)

    sharpe_ratio_SPY = np.sqrt(sf)*(avg_daily_ret_SPY-rfr)/std_daily_ret_SPY

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date.date(), end_date.date())
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    
    
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    
    orders = pd.read_csv( orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    symb = list(set(orders["Symbol"].unique()))
    
    # In the template, instfead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = orders.index[0]
    
    end_date = orders.index[-1]
    portvals = ut.get_data(symb, pd.date_range(start_date, end_date))

    cash = start_val
    values_shares = 0
    shares_owned = dict.fromkeys(symb, 0)

    portvals["val"] = np.nan
    for date,x in portvals.iterrows():
        #print("---")
        orders_date =  orders[ orders.index == date ]
        
        shares_owned_temp = shares_owned.copy()
        cash_temp = cash
        values_shares_temp = values_shares
        
        
        for i,order in orders_date.iterrows():
            action = order["Order"]
            nbShares = order["Shares"]
            stockSymbol = order["Symbol"]
            if action == 'BUY':
                cash_temp -= portvals[stockSymbol][date] * nbShares
                shares_owned_temp[stockSymbol] +=  nbShares
            if action == 'SELL':
                cash_temp += portvals[stockSymbol][date] * nbShares    
                shares_owned_temp[stockSymbol] -=  nbShares
        
          
        longs = 0
        shorts = 0
        values_shares_temp_cumul=0
        
        for stockSymbol_dict,nbShare_dict in shares_owned_temp.iteritems():
            
            stock_price = portvals[stockSymbol_dict][date]
            values_shares_temp = stock_price*nbShare_dict
            values_shares_temp_cumul += values_shares_temp
            if nbShare_dict > 0:
                longs += values_shares_temp
                
            else:
                shorts += values_shares_temp
                

        
        leverage = (longs + abs(shorts)) / (longs - abs(shorts) + cash_temp)

       
        if True:#leverage < 20000000:
            shares_owned = shares_owned_temp #uddate shares owned
            cash= cash_temp
            values_shares = values_shares_temp_cumul
        else:
            values_shares_temp_cumul=0
            #print("Leverage above two - trade of the day rejected")        
            #Do not update because trade cancelled
            #update value of previously owned shares without current price
            for stockSymbol_dict,nbShare_dict in shares_owned.iteritems():
                stock_price = portvals[stockSymbol_dict][date]
                values_shares_temp = stock_price*nbShare_dict
                values_shares_temp_cumul += values_shares_temp
                
            values_shares=values_shares_temp_cumul

        portvals["val"][date] =  cash + values_shares 

    portvals = portvals[['val']]    

    return portvals

def assess_portfolio(portvals, rfr=0.0, sf=252.0, \
    gen_plot=True):

    cr = (portvals[-1]-portvals[0]) / portvals[0]  #last -first /first
    
    daily_return = (portvals[1:].values/portvals[:-1].values) - 1  #difference between end of day and beginning of day

    adr = daily_return.mean()
    sddr = daily_return.std(ddof=1)
    sr = np.sqrt(sf)*(adr-rfr)/sddr

    return cr, adr, sddr, sr

def writeLine(orderFile,stockName,date,action,nb):
        line=[date.strftime("%Y-%m-%d")]
        line.append(stockName)
        line.append(action)
        line.append(str(nb)+"\n")     
        line2 = ",".join(line)
        orderFile.write(line2)

if __name__=="__main__":
    test_code(verb = False)
