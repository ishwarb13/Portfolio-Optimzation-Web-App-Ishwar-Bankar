import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def optimize_portfolio():


    stocks = pd.read_csv(f'C:/Users/kship/Documents/IB/predictions/results/SVR_Predictions.csv')
    stocks.drop(columns=['Unnamed: 0'], inplace=True)
    
    com=pd.read_csv(f'C:/Users/kship/Documents/IB/predictions/results/SVR_Predictions_C.csv')
    com.drop(columns=['Unnamed: 0'],inplace=True)

    combined = pd.concat([stocks, com], axis=1)
    combined.dropna(axis=0,inplace=True)

    def StockReturnsComputing(StockPrice, Rows, Columns):
        
        import numpy as np
        
        StockReturn = np.zeros([Rows-1, Columns])
        for j in range(Columns):        # j: Assets
            for i in range(Rows-1):     # i: Daily Prices
                StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100

        return StockReturn

    assetLabels = combined.columns.tolist()
    print('Asset labels of k-portfolio 1: \n', assetLabels)

    #read asset prices data
    StockData = combined.iloc[0:, 0:]

    #compute asset returns
    arStockPrices = np.asarray(StockData)
    [Rows, Cols]=arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    #set precision for printing results
    np.set_printoptions(precision=3, suppress = True)

    #compute mean returns and variance covariance matrix of returns
    meanReturns = np.mean(arReturns, axis = 0)
    covReturns = np.cov(arReturns, rowvar=False)
    print('\nMean Returns:\n', meanReturns)
    print('\nVariance-Covariance Matrix of Returns:\n', covReturns)


    from scipy import optimize 

    def MaximizeSharpeRatioOptmzn(MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
        
        # define maximization of Sharpe Ratio using principle of duality
        def  f(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
            funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T) )
            funcNumer = np.matmul(np.array(MeanReturns),x.T)-RiskFreeRate
            func = -(funcNumer / funcDenomr)
            return func

        #define equality constraint representing fully invested portfolio
        def constraintEq(x):
            A=np.ones(x.shape)
            b=1
            constraintVal = np.matmul(A,x.T)-b 
            return constraintVal
        
        
        #define bounds and other parameters
        xinit=np.repeat(0.33, PortfolioSize)
        cons = ({'type': 'eq', 'fun':constraintEq})
        lb = 0
        ub = 1
        bnds = tuple([(lb,ub) for x in xinit])\
        
        #invoke minimize solver
        opt = optimize.minimize (f, x0 = xinit, args = (MeanReturns, CovarReturns,\
                                RiskFreeRate, PortfolioSize), method = 'SLSQP',  \
                                bounds = bnds, constraints = cons, tol = 10**-3)
        
        return opt

    portfolioSize = len(combined.columns)

    #set risk free asset rate of return
    Rf=3  # April 2019 average risk  free rate of return in USA approx 3%
    annRiskFreeRate = Rf/100

    #compute daily risk free rate in percentage
    r0 = (np.power((1 + annRiskFreeRate),  (1.0 / 360.0)) - 1.0) * 100 
    print('\nRisk free rate (daily %): ', end="")
    print ("{0:.3f}".format(r0)) 

    #initialization
    xOptimal =[]
    minRiskPoint = []
    expPortfolioReturnPoint =[]
    maxSharpeRatio = 0

    #compute maximal Sharpe Ratio and optimal weights
    result = MaximizeSharpeRatioOptmzn(meanReturns, covReturns, r0, portfolioSize)
    xOptimal.append(result.x)

        
    #compute risk returns and max Sharpe Ratio of the optimal portfolio   
    xOptimalArray = np.array(xOptimal)
    Risk = np.matmul((np.matmul(xOptimalArray,covReturns)), np.transpose(xOptimalArray))

    expReturn = np.matmul(np.array(meanReturns),xOptimalArray.T)
    annRisk =   np.sqrt(Risk*251) 
    annRet = 251*np.array(expReturn) 
    maxSharpeRatio = (annRet-Rf)/annRisk 

    #set precision for printing results
    np.set_printoptions(precision=3, suppress = True)
    annRet=  annRet[0]
    annRisk =annRisk[0][0]

    #display results
    print('Maximal Sharpe Ratio: ', maxSharpeRatio, '\nAnnualized Risk (%):  ', \
        annRisk, '\nAnnualized Expected Portfolio Return(%):  ', annRet)
    print('\nOptimal weights (%):\n',  xOptimalArray.T*100 )

    
    return xOptimalArray, assetLabels, annRisk, annRet

        