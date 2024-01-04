from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


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
