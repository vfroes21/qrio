import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import riskfolio as rp

warnings.filterwarnings("ignore")

yf.pdr_override()
pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2016-01-01'
end = '2019-12-30'

# Tickers of assets
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
                  'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                            'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
assets.sort()

# Downloading data
data = yf.download(assets, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets

# Calculating returns

Y = data[assets].pct_change().dropna()
print(Y.head())

carteira = rp.Portfolio(returns=Y)

method_mu = 'hist'
method_cov = 'hist'

carteira.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

model = 'classic'   # ou mark
rm = 'MV'       # risk metric
obj = 'Sharpe'    # metrica de otimizacao
hist = True
rf = 0
l = 0      # representa funcao aversao a risco, complicado inicialmente

w = carteira.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)     # W comes from weight
