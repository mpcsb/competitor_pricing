# -*- coding: utf-8 -*-
"""
offers: value € per product

"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import zscore
import scipy
import pandas as pd

product_id = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
offers = np.array([10.0, 10.1, 10.3, 9.6, 10.7, 10.0, 9.9, 10.2, 9.7, 9.9, 20, 21, 19, 20, 18])
status = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
region = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1])


df = pd.DataFrame()
df['offers'] = offers
df['product_id'] = product_id
normalized_offers = df.groupby(['product_id']).offers.transform(lambda x : zscore(x, ddof=1))
normalized_offers -= min(normalized_offers) - 0.01



plt.scatter(status, normalized_offers, c=region)

#%%


with pm.Model() as m1:
    alpha = pm.Normal('alpha', mu=0, sigma=1)    
    out = pm.Normal('out', mu=alpha * offers, sigma=0.1, observed=status)
    trace = pm.sample(tune=1000, draws=1000, chains=4)
    
pm.traceplot(trace) 
plt.show()
sm = pm.summary(trace)

with m1:
    pp = pm.sample_posterior_predictive(trace)['out'].mean(axis=0)
    
plt.scatter(pp, status)
plt.xlabel('predicted')
plt.ylabel('status of offers')
plt.plot([0,1],[0,1], color='red', linestyle='dashed')
plt.show()

#%%

with pm.Model() as m2:
    beta = pm.Normal('beta', mu=0, sigma=1, shape=len(set(region)))
    alpha = pm.Normal('alpha', mu=0, sigma=1)   
    
    out = pm.Normal('out', mu=alpha * offers + beta[region], sigma=0.2, observed=status)
    trace = pm.sample(tune=1000, draws=1000, chains=4)
    
pm.traceplot(trace) 
plt.show()
sm = pm.summary(trace)

with m2:
    pp = pm.sample_posterior_predictive(trace)['out'].mean(axis=0)
    
plt.scatter(pp, status)
plt.xlabel('predicted')
plt.ylabel('status of offers')
plt.plot([0,1],[0,1], color='red', linestyle='dashed')
plt.show()

#%%

with pm.Model() as m3:
    rv_region = pm.Normal('rv_region', mu=0, sigma=1, shape=len(set(region)))
    rv_prod = pm.Normal('rv_prod', mu=0, sigma=1, shape=len(set(product_id)))
    alpha = pm.Normal('alpha', mu=0, sigma=1)   
    
    out = pm.Normal('out', mu=alpha * offers + rv_region[region] + rv_prod[product_id]
                    , sigma=0.2, observed=status)
    trace = pm.sample(tune=1000, draws=1000, chains=4)
    
pm.traceplot(trace) 
plt.show()
sm = pm.summary(trace)

with m3:
    pp = pm.sample_posterior_predictive(trace)['out'].mean(axis=0)
    
plt.scatter(pp, status)
plt.xlabel('predicted')
plt.ylabel('status of offers')
plt.plot([0,1],[0,1], color='red', linestyle='dashed')
plt.show()


#%% log linear

with pm.Model() as m4:
    rv_region = pm.Normal('rv_region', mu=0, sigma=1, shape=len(set(region)))
    rv_prod = pm.Normal('rv_prod', mu=0, sigma=1, shape=len(set(product_id)))
    alpha = pm.Normal('alpha', mu=0, sigma=1)   
    
    mu = alpha * np.log(offers) +  rv_region[region] + rv_prod[product_id]
    out = pm.Normal('out', mu=mu
                    , sigma=0.2, observed=status)
    trace = pm.sample(tune=1000, draws=1000, chains=4)
    
  
pm.traceplot(trace) 
plt.show()
sm = pm.summary(trace)

with m4:
    pp = pm.sample_posterior_predictive(trace)['out'].mean(axis=0)
    
plt.scatter(pp, status)
plt.xlabel('predicted')
plt.ylabel('status of offers')
plt.plot([0,1],[0,1], color='red', linestyle='dashed')
plt.show() 


#%% 

with pm.Model() as m5:
    rv_region = pm.Normal('rv_region', mu=0, sigma=1, shape=len(set(region)))
    rv_prod = pm.Normal('rv_prod', mu=0, sigma=1, shape=len(set(product_id)))
    rv_value = pm.Normal('rv_value', mu=0, sigma=1)   
    
    mu = rv_value * np.log(normalized_offers) +  rv_region[region] + rv_prod[product_id]
    
    sigma = pm.HalfNormal('sigma', sigma=0.2)
    y = pm.Normal('y', mu=mu
                    , sigma=sigma, observed=status)
    trace = pm.sample(tune=1000, draws=5000, chains=1)
    
  
pm.traceplot(trace) 
plt.show()
sm = pm.summary(trace)

with m5:
    pp = pm.sample_posterior_predictive(trace)['y'].mean(axis=0)
    
plt.scatter(pp, status, alpha=0.3)
plt.xlabel('predicted')
plt.ylabel('status of offers')
plt.plot([0,1],[0,1], color='red', linestyle='dashed')
plt.show() 

# pm.model_to_graphviz(m5)

# p = scipy.special.expit(pp)

# plt.scatter(p, status, alpha=0.3)
# plt.xlabel('predicted')
# plt.ylabel('status of offers')
# plt.plot([0,1],[0,1], color='red', linestyle='dashed')
# plt.axvline(0.5)
# plt.show() 

#%%

