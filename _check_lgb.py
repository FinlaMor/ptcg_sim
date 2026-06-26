import lightgbm as lgb
import inspect
print(lgb.__version__)
print(inspect.signature(lgb.train))
