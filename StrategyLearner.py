import numpy as np
import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut

class StrategyLearner(object):

    def __init__(self, verbose = False):
        self.verbose = verbose
        self.ql = ql.QLearner(num_states=30000, num_actions=3, dyna=0, rar=0.8, radr=0.99)
        self.momentumWindow = 19
        self.rollingWindow = 20

    def addEvidence(self, symbol = "IBM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,1,1), sv = 10000): 
        
        self.symbol = symbol
        self.startVal = sv
        myFeatures = self.createFeature(self.momentumWindow,self.rollingWindow,sd,ed,symbol)
        self.thresholds = self.createTresholds(myFeatures.ix[:,1:])
        self.TrainQL(myFeatures)
        
        
    def TrainQL(self, myFeatures):

        stockName = self.symbol
        PortfolioValues = np.full((myFeatures.shape[0]), 0.0)
          
        if myFeatures.shape[0] > self.rollingWindow:
            
            for i in range(0, 100):
    
                PortfolioValues[0] = self.startVal
                valueShares = self.startVal
                sdate = max(self.rollingWindow,self.momentumWindow)
                myPosition = 1
                NbShares = 0
    
                state = self.getState(myFeatures.ix[sdate][1:],myPosition)
                action = self.ql.querysetstate(state)
    
                for day in range(sdate, myFeatures.shape[0]):
                    stockPrice = myFeatures[stockName][day]
    
                    myPosition, NbShares, valueShares, portval = self.getNewValue(myPosition,NbShares,  valueShares,stockPrice,action)
    
                    PortfolioValues[day] = portval
                    
                    rew = (PortfolioValues[day] / PortfolioValues[day - 1]) - 1
                    if rew < 0:
                        reward = 50 * rew
                    else:
                        reward = 10 * rew 
    
                    state = self.getState(myFeatures.ix[day][1:],myPosition)
                    action = self.ql.query(state, reward)
                    
                print i, PortfolioValues[myFeatures.shape[0]-1]


    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1),sv = 10000):
        self.symbol = symbol
        myFeatures = self.createFeature(self.momentumWindow, self.rollingWindow, sd, ed, symbol)
        trades = self.getOrders(myFeatures)
        if myFeatures.shape[0] > max(self.momentumWindow, self.rollingWindow):
            trades = self.getOrders(myFeatures)
        else:
            trades =  pd.DataFrame(0,index=myFeatures.index, columns=['Actions'])
            
        return trades
    
    def getOrders(self, myFeatures):
#         deliverable1 = open("orderStrat.csv", 'w')   
#         deliverable1.write("Date,Symbol,Order,Shares\n")
        
        df_trades = pd.DataFrame(0,index=myFeatures.index, columns=['Actions'])
        NbShares = 0
        
        BUY = 0
        SELL = 1
        
        myPosition = 1
        
        for day in range(max(self.rollingWindow,self.momentumWindow), myFeatures.shape[0]-1):

            state = self.getState(myFeatures.ix[day][1:],myPosition) 
            
            action = self.ql.querysetstate(state)

            if action == BUY and NbShares < 100:
                df_trades['Actions'][day] = 100
                #self.writeLine(deliverable1,symbol,myFeatures.index[date],"BUY",100)
                NbShares += 100
                myPosition += 1
            elif action == SELL and NbShares > -100:
                df_trades['Actions'][day] = -100
                #self.writeLine(deliverable1,symbol,myFeatures.index[date],"SELL",100)
                NbShares -= 100
                myPosition -= 1

        if NbShares == 100:
            df_trades['Actions'][-1] = -100
            #self.writeLine(deliverable1,symbol,myFeatures.index[-1],"SELL",100)
        elif NbShares == -100:
            df_trades['Actions'][-1] = 100
            #self.writeLine(deliverable1,symbol,myFeatures.index[-1],"BUY",100)

        #deliverable1.close() 
        return df_trades
    
    def createFeature(self,momentumWindow,featurewindow,start_date,end_date,stockName):
        
        portvals = ut.get_data([stockName], pd.date_range(start_date, end_date))
        rm_SPY = pd.rolling_mean(portvals[stockName], window=featurewindow)
        rstd_SPY = pd.rolling_std(portvals[stockName], window=featurewindow)
        
        sma = portvals[stockName]/rm_SPY-1
        bb_value = (portvals[stockName] - rm_SPY)/(2 * rstd_SPY)
    
        momentum = (portvals[(momentumWindow):][stockName].values/portvals[:-(momentumWindow)][stockName].values) - 1
        
        daily_return = (portvals[1:][stockName].values/portvals[:-1][stockName].values) - 1  #difference between end of day and beginning of day
        daily_return = np.insert(daily_return, 0, 0)
        
        volatility = pd.rolling_std(daily_return, window=featurewindow)

        portvals['bollinger'] = bb_value
        portvals['sma'] = sma
        portvals['momentum'] = np.nan
        portvals['volatility'] = np.nan
        portvals.loc[(momentumWindow):,'momentum'] = momentum
        portvals.loc[:,'volatility'] = volatility
        
        portvals['bollinger'] = (portvals['bollinger']-portvals['bollinger'].mean())/portvals['bollinger'].std()
        portvals['volatility'] = (portvals['volatility']-portvals['volatility'].mean())/portvals['volatility'].std()
        portvals['momentum'] = (portvals['momentum']-portvals['momentum'].mean())/portvals['momentum'].std()
        portvals['sma'] = (portvals['sma']-portvals['sma'].mean())/portvals['sma'].std()
        
        feat = portvals.fillna(0)
    
        trainX = feat.drop(["SPY","sma"],1)

        return trainX
    
    def getState(self, df_feat, myPosition):
        featState = 0
        for feat in range(0,df_feat.shape[0]):
            for i in range(0, self.binNb):
                if df_feat[feat] <= self.thresholds[i,feat]:
                    featState += i * pow(10,feat)
                    break
        return featState + myPosition * pow(10,df_feat.shape[0]-1)
    
    def createTresholds(self, myFeatures):
        thresholds = np.zeros((10,myFeatures.shape[1]))
        self.binNb = 10
        stepsize = myFeatures.shape[0] / self.binNb 
        for feat in range(myFeatures.shape[1]):
            for i in range(0, self.binNb):
                thresholds[i, feat] = np.sort(myFeatures.ix[:,feat].values)[(i+1)*stepsize-1]
        return thresholds
    
    
    def getNewValue(self, myPosition, NbShares, valueShares, stockPrice, action):
        BUY = 0
        SELL = 1
        
        if action == SELL and NbShares > -100:
            NbShares -= 100
            myPosition -= 1
            valueShares += 100 * stockPrice
        elif action == BUY and NbShares < 100:
            NbShares += 100
            myPosition += 1
            valueShares -= 100 * stockPrice
            
        return myPosition , NbShares, valueShares, valueShares + NbShares * stockPrice
    
#     def writeLine(self,orderFile,stockName,date,action,nb):
#         line=[date.strftime("%Y-%m-%d")]
#         line.append(stockName)
#         line.append(action)
#         line.append(str(nb)+"\n")     
#         line2 = ",".join(line)
#         orderFile.write(line2)

if __name__=="__main__":
    print "One does not simply think up a strategy"
