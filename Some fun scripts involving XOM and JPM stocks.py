# %% [markdown]
# # Some fun scripts involving XOM and JPM stocks
# **Owner**: Thomas De Massari    
# **Linkedin**: https://www.linkedin.com/in/thomasdemassari/  
# **GitHub**: https://github.com/thomasdemassari/     
# **Course**: Data Science Summer School 2023, University of Trento  
# 
# I wrote these short scripts to apply the fundamental concepts from the Data Science Summer School 2023 course.

# %% [markdown]
# ## Libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import yfinance as yf

# %% [markdown]
# ## Data upload

# %%
xom = yf.download("XOM", start = "2012-01-01", end = "2022-12-31")
jpm = yf.download("JPM", start = "2012-01-01", end = "2022-12-31")
sp500 = yf.download("^SPX", start = "2012-01-01", end = "2022-12-31")

# Needed for get Date (without hours)
xom.reset_index(inplace=True)
jpm.reset_index(inplace=True)
sp500.reset_index(inplace=True)

xom['Date'] = xom['Date'].dt.date
jpm['Date'] = jpm['Date'].dt.date
sp500['Date'] = sp500['Date'].dt.date

# %% [markdown]
# ## Graphical analysis

# %% [markdown]
# ### XOM

# %%
date = xom["Date"]
price = xom["Close"]
volume = xom["Volume"]

# PRICE
fig, ax1 = plt.subplots(figsize = (15, 8))
ax1.plot(date, price, color = "black", label = "Price")
ax1.set_xlabel("Days")
ax1.set_ylabel("Price ($)")
ax1.set_xticks(date[::70])
ax1.set_xticklabels(date[::70], rotation=90)

# AVG and Median
price_avg = price.mean()
price_median = price.median()
ax1.axhline(y = price_avg, color = "orange", linestyle = "-.", label = "Average price")
ax1.axhline(y = price_median, color = "green", linestyle = "-.", label = "Median price")

# MAX and MIN price
ax1.plot(date[price.idxmax()], price.max(), "go")
ax1.annotate(f"Maximum point ({round(price.max(), 2)} $)", (date[price.idxmax()], price.max()), textcoords = "offset points", xytext = (-6, -2), ha = "right", va = "bottom")

ax1.plot(date[price.idxmin()], price.min(), "ro")
ax1.annotate(f"Minimum point ({round(price.min(), 2)} $)", (date[price.idxmin()], price.min()), textcoords = "offset points", xytext = (10, -5), ha = 'left', va = "bottom")

# Fist and last day
x_first = date[0]
y_first = price[0]
ax1.plot(x_first, y_first, "ko")
ax1.annotate("First day", (x_first, y_first), textcoords = "offset points", xytext = (-30, +20), va = "top")

x_last = date[len(date)-1]
y_last = price[len(price)-1]
ax1.plot(x_last, y_last, "ko")
ax1.annotate("Last day", (x_last, y_last), textcoords = "offset points", xytext = (+3, -5), va = "top")

ax1.grid(True)

# VOLUME
ax2 = ax1.twinx()
ax2.bar(date, volume, color = "grey", alpha = 0.3, label = "Volume")
ax2.set_ylabel("Volume", color = "black")
ax2.tick_params(axis = "y", labelcolor = "black")

# AVG e Median
volume_average = volume.mean()
volume_median = volume.median()
ax2.axhline(y = volume_average, color = "purple", linestyle = "-.", label = "Average volume")
ax2.axhline(y = volume_median, color = "blue", linestyle = "-.", label = "Median volume")

fig.legend(loc = "upper left", bbox_to_anchor = (0,1), bbox_transform = ax1.transAxes)

plt.title("XOM Stock price and volume trend")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### JPM

# %%
date = jpm["Date"]
price = jpm["Close"]
volume = jpm["Volume"]

# PRICE
fig, ax1 = plt.subplots(figsize = (15, 8))
ax1.plot(date, price, color = "black", label = "Price")
ax1.set_xlabel("Days")
ax1.set_ylabel("Price ($)")
ax1.set_xticks(date[::70])
ax1.set_xticklabels(date[::70], rotation=90)

# AVG and Median
price_avg = price.mean()
price_median = price.median()
ax1.axhline(y = price_avg, color = "orange", linestyle = "-.", label = "Average price")
ax1.axhline(y = price_median, color = "green", linestyle = "-.", label = "Median price")

# MAX and MIN price
ax1.plot(date[price.idxmax()], price.max(), "go")
ax1.annotate(f"Maximum point ({round(price.max(), 2)} $)", (date[price.idxmax()], price.max()), textcoords = "offset points", xytext = (-6, -2), ha = "right", va = "bottom")

ax1.plot(date[price.idxmin()], price.min(), "ro")
ax1.annotate(f"Minimum point ({round(price.min(), 2)} $)", (date[price.idxmin()], price.min()), textcoords = "offset points", xytext = (10, -5), ha = 'left', va = "bottom")

# Fist and last day
x_first = date[0]
y_first = price[0]
ax1.plot(x_first, y_first, "ko")
ax1.annotate("First day", (x_first, y_first), textcoords = "offset points", xytext = (-30, +20), va = "top")

x_last = date[len(date)-1]
y_last = price[len(price)-1]
ax1.plot(x_last, y_last, "ko")
ax1.annotate("Last day", (x_last, y_last), textcoords = "offset points", xytext = (+3, -5), va = "top")

ax1.grid(True)

# VOLUME
ax2 = ax1.twinx()
ax2.bar(date, volume, color = "grey", alpha = 0.3, label = "Volume")
ax2.set_ylabel("Volume", color = "black")
ax2.tick_params(axis = "y", labelcolor = "black")

# AVG e Median
volume_average = volume.mean()
volume_median = volume.median()
ax2.axhline(y = volume_average, color = "purple", linestyle = "-.", label = "Average volume")
ax2.axhline(y = volume_median, color = "blue", linestyle = "-.", label = "Median volume")


fig.legend(loc = "upper left", bbox_to_anchor = (0,1), bbox_transform = ax1.transAxes)

plt.title("JPM Stock price and volume trend")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### SP500

# %%
date = sp500["Date"]
price = sp500["Close"]

plt.figure(figsize=(15, 8)) 
plt.plot(date, price, color = "red")

plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.xticks(date[::70], rotation=90)

# AVG and Median
price_avg = price.mean()
price_median = price.median()
plt.axhline(y = price_avg, color = "orange", linestyle = "-.", label = "Average price")
plt.axhline(y = price_median, color = "green", linestyle = "-.", label = "Median price")

# MAX and MIN price
plt.plot(date[price.idxmax()], price.max(), "go")
plt.annotate(f"Maximum point ({round(price.max(), 2)} $)", (date[price.idxmax()], price.max()), textcoords = "offset points", xytext = (-6, -2), ha = "right", va = "bottom")

plt.plot(date[price.idxmin()], price.min(), "ro")
plt.annotate(f"Minimum point ({round(price.min(), 2)} $)", (date[price.idxmin()], price.min()), textcoords = "offset points", xytext = (10, -5), ha = 'left', va = "bottom")

# Fist and last day
x_first = date[0]
y_first = price[0]
plt.plot(x_first, y_first, "ko")
plt.annotate("First day", (x_first, y_first), textcoords = "offset points", xytext = (-30, +20), va = "top")

x_last = date[len(date)-1]
y_last = price[len(price)-1]
plt.plot(x_last, y_last, "ko")
plt.annotate("Last day", (x_last, y_last), textcoords = "offset points", xytext = (+3, -5), va = "top")

plt.grid(True)

plt.title("SP500 price trend")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Markowitz (1952) - Portfolio Selection

# %% [markdown]
# ### Function
# The function takes as input:
# - a tuple containing, for each position, a Pandas DataFrame that includes stock prices;
# - a tuple containing the column numbers where the stock prices (not returns) are located.
# 
# It returns a dictionary with two Pandas DataFrame: one with returns, standard deviation, and portfolio weights ("Result") and the other one with correlations ("Correlations"). If the parameter 'chart' is set to True, it also plots the chart; by default, it is set to False.

# %%
def PortfolioSelection(securities, cols_price, chart = False):
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import math
        import random
        from datetime import datetime

        daily_returns = list()                                          # Where I will save daily returns
        returns = list()                                                # Where I will save average returns
        sds = list()                                                    # Where I will save standard deviations of daily returns

        for i in range(len(securities)):
            prices = securities[i].iloc[:, cols_price[i]]               # Stock prices of security i
            daily_return = (prices.pct_change()).iloc[1:]               # Daily return of security i
            daily_returns.append(daily_return)                          
            returns.append(np.mean(daily_return))                       # Average return 

            sd = np.std(daily_return)                                   # Standard deviation of security i
            sds.append(sd)                                             


        port_returns = list()                                           # Portfolio returns
        port_sds = list()                                               # Portfolio standard deviations
        saved_weigths = list()                                          # Weights of securities
        
        # Temporary variables
        reps = 0 
        portfolio_returns_tmp = 0
        portfolio_sds_tmp = 0

        correlations = np.zeros((len(securities), len(securities)))

        while reps <= 1000:
            weights = list()                                            # Weights of portfolio j 
            list_tmp = list()                                           # Where I will save temporarily weights in each cycle 

            # Randomly choose weights (0 <= w <= 1). The sum must be 1.
            for j in range(len(securities) - 1):
                # np.random.seed(226091)                                # (226091 is my Student ID number at University of Trento)
                n_random = random.uniform(0, 1)                         
                weights.append(n_random)
                list_tmp.append(round(n_random*100, 3))          

            last_weights = 1 - sum(weights)
            weights.append(last_weights)
            list_tmp.append(round(last_weights*100, 3))
            saved_weigths.append(list_tmp)

            # Portfolio returns and standard deviations
            for i in range(len(securities)):       
                # Portfolio returns                        
                portfolio_returns_tmp = portfolio_returns_tmp + weights[i] * returns[i] 

                # Portfolio standard deviations
                for j in range(len(securities)):
                    cor = float(np.corrcoef(daily_returns[i], daily_returns[j])[0, 1])
                    tmp = weights[i] * weights[j] * sds[i] * sds[j] * cor
                    portfolio_sds_tmp = portfolio_sds_tmp + tmp 

                    correlations[i][j] = cor
                    correlations[j][i] = cor
                
                correlations[i][i] = 1

            port_sds.append(math.sqrt(portfolio_sds_tmp) * 100)
            port_returns.append(portfolio_returns_tmp * 100)

            portfolio_returns_tmp = 0
            portfolio_sds_tmp = 0

            reps = reps + 1

        result = pd.DataFrame({"Returns (%)": port_returns, "Standard Deviation (%)": port_sds, "Weights (%)": saved_weigths})
        result = result.sort_values(by="Standard Deviation (%)")
        result = result.reset_index()

        # Chart
        if chart == True:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            min_sd = np.argmin(port_sds)
            index_min_r =  port_returns[min_sd]

            facecolors_list = list()
            for z in range(len(port_returns)):
                if(port_returns[z] >= index_min_r):
                    facecolors_list.append("black")
                else:
                    facecolors_list.append("none")

            plt.figure(figsize=(15, 6))
            plt.scatter(port_sds, port_returns, color = "black", facecolors = facecolors_list, linewidths=0.5)

            plt.title(f"Efficient frontier (printed on: {date})")
            plt.xlabel("Standard deviation (%)")
            plt.ylabel("Returns (%)")

            plt.axhline(y = round(index_min_r, 4), color = "k", linestyle = "--")

            plt.show()


        # Correlations between securities
        titles_names = [f"Security {index+1}" for index in range(len(securities))]
        correlations = pd.DataFrame(correlations, index=titles_names, columns=titles_names)
        
        # Result 
        result_dict = {
            "Result": result,
            'Correlations': correlations
        }

        return result_dict
    
    except:
        raise Exception("Something went wrong.")

# %% [markdown]
# ### Implementation

# %%
secs = (xom, jpm)
ncols = (4, 4)

xom_jpm = PortfolioSelection(secs, ncols, chart = True)
print(xom_jpm["Result"])
print(xom_jpm["Correlations"])

# %% [markdown]
# ## Sharpe (1963) - A Simplified Model for Portfolio Analysis

# %%
returns_jpm = (jpm["Close"]).pct_change().iloc[1:] 
returns_xom = (xom["Close"]).pct_change().iloc[1:] 
returns_sp500 = (sp500["Close"]).pct_change().iloc[1:] 

returns_sp500_cost = sm.add_constant(returns_sp500)

# XOM
sharpe63_xom = sm.OLS(returns_xom, returns_sp500_cost)
print((sharpe63_xom.fit()).summary())
# Chart 
plt.figure(figsize = (10, 6))
sns.regplot(x = returns_sp500, y = returns_xom, ci = None, line_kws = {"color": "red"})
plt.xlabel("S&P 500")
plt.ylabel("XOM")
plt.grid(True)
plt.tight_layout()
plt.show()

# JPM
sharpe63_jpm = sm.OLS(returns_jpm, returns_sp500_cost)
print((sharpe63_jpm.fit()).summary())
# Chart 
plt.figure(figsize = (10, 6))
sns.regplot(x = returns_sp500, y = returns_jpm, ci = None, line_kws = {"color": "red"})
plt.xlabel("S&P 500")
plt.ylabel("JPM")
plt.grid(True)
plt.tight_layout()
plt.show()


