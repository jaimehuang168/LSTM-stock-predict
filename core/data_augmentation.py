import math
import numpy as np
import pandas as pd
import talib

class DataAugmentation():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename):
        self.dataframe = pd.read_csv(filename)
        #print(talib.get_function_groups()['Momentum Indicators'])
        
    def getIndicators(self):
        date = self.dataframe.get('Date').values
        open = self.dataframe.get('Open').values
        high = self.dataframe.get('High').values
        low = self.dataframe.get('Low').values
        close = self.dataframe.get('Close').values
        volume = self.dataframe.get('Volume').values
       
        # calculate indicators from raw data
        timeperiod = 50
        adx = talib.ADX(high, low, close, timeperiod)
        adxr = talib.ADXR(high, low, close, timeperiod)
        aroonosc = talib.AROONOSC(high, low, timeperiod)
        bop = talib.BOP(open, high, low, close)
        cci = talib.CCI(high, low, close, timeperiod)
        cmo = talib.CMO(close, timeperiod)
        dx = talib.DX(high, low, close, timeperiod)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod)
        minus_dm = talib.MINUS_DM(high, low, timeperiod)
        mom = talib.MOM(close, timeperiod)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod)
        plus_dm = talib.PLUS_DM(high, low, timeperiod)
        roc = talib.ROC(close, timeperiod)
        rocp = talib.ROCP(close, timeperiod)
        rocr = talib.ROCR(close, timeperiod)
        rocr100 = talib.ROCR100(close, timeperiod)
        rsi = talib.RSI(close, timeperiod)
        trix = talib.TRIX(close, timeperiod)
        willr = talib.WILLR(high, low, close, timeperiod)
        
        
        

        output = pd.DataFrame({'Date':date, 'Open':open, 'High':high, 'Low':low, "Close":close,
                               'Volume':volume, 'Adx':adx, 'Adxr':adxr, 'Aroonosc':aroonosc,'Cci':cci,
                               'Bop':bop, 'Cmo':cmo, 'Dx':dx, 'Minus_di':minus_di,
                               'Minus_dm':minus_dm, 'Mom':mom, 'Plus_di':plus_di, 'Plus_dm':plus_dm,
                               'Roc':roc, 'Rocp':rocp, 'Rocr':rocr, 'Rocr100':rocr100, 'Rsi':rsi,
                               'Trix':trix, 'Willr':willr})
        
        # should avoid nan part for training
        for key, value in output.items():
            # if key != 'Date' and np.isnan(value[0]):
                # output[key].fillna((output[key].mean()), inplace=True)
            if key != 'Date':
                nans = 0
                for v in value:
                    if np.isnan(v):
                        nans += 1
                # print(key, nans)
        #print(len(output['Adx']))
        # output.to_csv('sp500_beta.csv', index=False)
        output.to_csv('sp500_2018end_beta.csv', index=False)
        
        


# d = DataAugmentation('sp500.csv')
d = DataAugmentation('sp500_2018end.csv')
d.getIndicators()
