import talib as tb
import numpy as np


# define pivot variables for easy use
def technical(df):
    open = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    # define the technical analysis matrix
    retn = np.array([
        tb.MA(close, timeperiod=60),                                        # 1
        tb.MA(close, timeperiod=120),                                       # 2

        tb.ADX(high, low, close, timeperiod=14),                            # 3
        tb.ADXR(high, low, close, timeperiod=14),                           # 4

        tb.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0],    # 5
        tb.RSI(close, timeperiod=14),                                       # 6

        tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0],  # 7
        tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1],  # 8
        tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2],  # 9

        tb.AD(high, low, close, volume),                                    # 10
        tb.ATR(high, low, close, timeperiod=14),                            # 11

        tb.HT_DCPERIOD(close),                                              # 12

        tb.CDL2CROWS(open, high, low, close),                               # 13
        tb.CDL3BLACKCROWS(open, high, low, close),                          # 14
        tb.CDL3INSIDE(open, high, low, close),                              # 15
        tb.CDL3LINESTRIKE(open, high, low, close),                          # 16
        tb.CDL3OUTSIDE(open, high, low, close),                             # 17
        tb.CDL3STARSINSOUTH(open, high, low, close),                        # 18
        tb.CDL3WHITESOLDIERS(open, high, low, close),                       # 19
        tb.CDLABANDONEDBABY(open, high, low, close, penetration=0),         # 20
        tb.CDLADVANCEBLOCK(open, high, low, close),                         # 21
        tb.CDLBELTHOLD(open, high, low, close),                             # 22
        tb.CDLBREAKAWAY(open, high, low, close),                            # 23
        tb.CDLCLOSINGMARUBOZU(open, high, low, close),                      # 24
        tb.CDLCONCEALBABYSWALL(open, high, low, close),                     # 25
        tb.CDLCOUNTERATTACK(open, high, low, close),                        # 26
        tb.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0),        # 27
        tb.CDLDOJI(open, high, low, close),                                 # 28
        tb.CDLDOJISTAR(open, high, low, close),                             # 29
        tb.CDLDRAGONFLYDOJI(open, high, low, close),                        # 30
        tb.CDLENGULFING(open, high, low, close),                            # 31
        tb.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0),       # 32
        tb.CDLEVENINGSTAR(open, high, low, close, penetration=0),           # 33
        tb.CDLGAPSIDESIDEWHITE(open, high, low, close),                     # 34
        tb.CDLGRAVESTONEDOJI(open, high, low, close),                       # 35
        tb.CDLHAMMER(open, high, low, close),                               # 36
        tb.CDLHANGINGMAN(open, high, low, close),                           # 37
        tb.CDLHARAMI(open, high, low, close),                               # 38
        tb.CDLHARAMICROSS(open, high, low, close),                          # 39
        tb.CDLHIGHWAVE(open, high, low, close),                             # 40
        tb.CDLHIKKAKE(open, high, low, close),                              # 41
        tb.CDLHIKKAKEMOD(open, high, low, close),                           # 42
        tb.CDLHOMINGPIGEON(open, high, low, close),                         # 43
        tb.CDLIDENTICAL3CROWS(open, high, low, close),                      # 44
        tb.CDLINNECK(open, high, low, close),                               # 45
        tb.CDLINVERTEDHAMMER(open, high, low, close),                       # 46
        tb.CDLKICKING(open, high, low, close),                              # 47
        tb.CDLKICKINGBYLENGTH(open, high, low, close),                      # 48
        tb.CDLLADDERBOTTOM(open, high, low, close),                         # 49
        tb.CDLLONGLEGGEDDOJI(open, high, low, close),                       # 50
        tb.CDLLONGLINE(open, high, low, close),                             # 51
        tb.CDLMARUBOZU(open, high, low, close),                             # 52
        tb.CDLMATCHINGLOW(open, high, low, close),                          # 53
        tb.CDLMATHOLD(open, high, low, close, penetration=0),               # 54
        tb.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0),       # 55
        tb.CDLMORNINGSTAR(open, high, low, close, penetration=0),           # 56
        tb.CDLONNECK(open, high, low, close),                               # 57
        tb.CDLPIERCING(open, high, low, close),                             # 58
        tb.CDLRICKSHAWMAN(open, high, low, close),                          # 59
        tb.CDLRISEFALL3METHODS(open, high, low, close),                     # 60
        tb.CDLSEPARATINGLINES(open, high, low, close),                      # 61
        tb.CDLSHOOTINGSTAR(open, high, low, close),                         # 62
        tb.CDLSHORTLINE(open, high, low, close),                            # 63
        tb.CDLSPINNINGTOP(open, high, low, close),                          # 64
        tb.CDLSTALLEDPATTERN(open, high, low, close),                       # 65
        tb.CDLSTICKSANDWICH(open, high, low, close),                        # 66
        tb.CDLTAKURI(open, high, low, close),                               # 67
        tb.CDLTASUKIGAP(open, high, low, close),                            # 68
        tb.CDLTHRUSTING(open, high, low, close),                            # 69
        tb.CDLTRISTAR(open, high, low, close),                              # 70
        tb.CDLUNIQUE3RIVER(open, high, low, close),                         # 71
        tb.CDLUPSIDEGAP2CROWS(open, high, low, close),                      # 72
        tb.CDLXSIDEGAP3METHODS(open, high, low, close)                      # 73
    ]).T
    return retn
