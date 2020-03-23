
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class Strategy006(IStrategy):
    """
    Strategy 006
    author@: Adam Phoenix

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy005
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "89": 0.001,
        "55": 0.005,
        "34": 0.03,
        "21": 0.05,
        "13": 0.08,
        "8": 0.13,
        "0": 0.08
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    # Optimal ticker interval for the strategy
    ticker_interval = '1m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['slowadx'] = ta.ADX(dataframe, 2)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # MACD the difference between an instrument's 26-day and 12-day exponential moving averages (EMA)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Minus Directional Indicator / Movement
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # RSI
        rsi = ta.RSI(dataframe)
        dataframe['rsi'] = rsi

        # EMA - Exponential Moving Average
        dataframe['ema2'] = ta.EMA(dataframe, timeperiod=2)
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (rsi - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=12, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_midband'] = bollinger['mid']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['fastk-previous'] = dataframe.fastk.shift(1)
        dataframe['fastd-previous'] = dataframe.fastd.shift(1)

        # Stoch
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Overlap Studies
        # ------------------------------------

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # Prod
            (
                (dataframe['rsi'] > 0)
                & (dataframe['rsi'] < 85)
                #& (dataframe['mfi'] > 10.0)
                & (dataframe['close'] > 0.00000200)
                & (dataframe['close'] < dataframe['ema13'])
                & (dataframe['bb_upperband'] > dataframe['ema8'])
                #& (dataframe['close'] > dataframe['bb_lowerband'])
                #& (dataframe['slowd'] > dataframe['slowk'])
                #& (dataframe['slowd'] > 0)
                #& (dataframe['fastd'] > dataframe['fastk'])
                #& (dataframe['fastd'] > 0)
                #& (dataframe['fisher_rsi'] < -0.94)
                & (dataframe['fisher_rsi'] < 0)
                #& (qtpylib.crossed_above(dataframe['htleadsine'], dataframe['htsine']))
                & (dataframe['htleadsine'] < 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # Prod
            (dataframe['fisher_rsi'] > 0)
            & (dataframe['minus_di'] > 0)
            & (dataframe['sar'] > dataframe['close'])
            & (
                (qtpylib.crossed_below(dataframe['htleadsine'], dataframe['htsine']))
                | (
                    (dataframe['fastk-previous'] < dataframe['fastd-previous'])
                    & (dataframe['fastd'] < dataframe['fastk'])
                )
                | (dataframe['bb_midband'] < dataframe['ema2'])
                # | (
                #     ((dataframe['fastk'] > 70) | (dataframe['fastd'] > 70)) &
                #     (dataframe['fastk-previous'] < dataframe['fastd-previous']) &
                #     (dataframe['close'] > dataframe['ema2'])
                # )
            )
            , 'sell'] = 1
        return dataframe
