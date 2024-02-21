from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType

""" cn stock
high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open_ = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)
target = Ref(close, -20) / close - 1
amount = Feature(FeatureType.AMOUNT)
isst = Feature(FeatureType.ISST)
pbmrq = Feature(FeatureType.PBMRQ)
pcfncfttm = Feature(FeatureType.PCFNCFTTM)
pctchg = Feature(FeatureType.PCTCHG)
pettm = Feature(FeatureType.PETTM)
preclose = Feature(FeatureType.PRECLOSE)
psttm = Feature(FeatureType.PSTTM)
turn = Feature(FeatureType.TURN)
"""

high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open_ = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
quote_volume = Feature(FeatureType.QUOTE_VOLUME)
trade_num = Feature(FeatureType.TRADE_NUM)
taker_buy_quote_asset_volume = Feature(FeatureType.TAKER_BUY_QUOTE_ASSET_VOLUME)
taker_buy_base_asset_volume = Feature(FeatureType.TAKER_BUY_BASE_ASSET_VOLUME)
circulating_supply = Feature(FeatureType.CIRCULATING_SUPPLY)
circulating_mcap = Feature(FeatureType.CIRCULATING_MCAP)
num_market_pairs = Feature(FeatureType.NUM_MARKET_PAIRS)
pct_diff_abs = Feature(FeatureType.PCT_DIFF_ABS)
total_mcap = Feature(FeatureType.TOTAL_MCAP)
turnover_rate = Feature(FeatureType.TURNOVER_RATE)
usd_price = Feature(FeatureType.USD_PRICE)
usd_price_pct = Feature(FeatureType.USD_PRICE_PCT)
usd_volume = Feature(FeatureType.USD_VOLUME)

EXPR_MAP = {
    "$high": "high",
    "$low": "low",
    "$volume": "volume",
    "$open": "open_",
    "$close": "close",
    "$quote_volume": "quote_volume",
    "$trade_num": "trade_num",
    "$taker_buy_base_asset_volume": "taker_buy_base_asset_volume",
    "$taker_buy_quote_asset_volume": "taker_buy_quote_asset_volume",
    "$circulating_supply": "circulating_supply",
    "$circulating_mcap": "circulating_mcap",
    "$num_market_pairs": "num_market_pairs",
    "$pct_diff_abs": "pct_diff_abs",
    "$total_mcap": "total_mcap",
    "$turnover_rate": "turnover_rate",
    "$usd_price": "usd_price",
    "$usd_price_pct": "usd_price_pct",
    "$usd_volume": "usd_volume",
}    