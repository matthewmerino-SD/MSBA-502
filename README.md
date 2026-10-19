# MSBA-502
This repo will contain the files used in our MSBA 502 project. Created by Matt Merino and Perry Fan

There's a running joke on Wall Street: quants know every advanced model, but at the end of the day they use linear regression. Meanwhile, investors pay a premium for "quant" and "smart" investment products built on complex machine learning.

As Man AHL's chief scientist Anthony Ledford put it, machine learning is "more robust than its predecessors but hardly revolutionary." We wanted to test that ourselves: do complex models actually beat simple ones?

Choosing a target: risk, not returns

We started by trying to predict stock returns and quickly found what the research says: returns are mostly noise (Gu, Kelly & Xiu, 2020). Every model we tried scored a negative R² on unseen data, and complex models overfit badly.

So we switched to predicting volatility: how much a stock's price will swing next month. Volatility is predictable, easy to explain to a client, and a core part of portfolio performance metrics like the Sharpe ratio.

What we did
Collected 15 years of data on 495 S&P 500 companies (87,864 stock-months, 2011 to 2026)
Built two sets of predictors:
Set A: 12 variables based only on each stock's price history (past volatility, beta, momentum, drawdown and others)
Set B: all 39 variables, adding the VIX, interest rates, company financials (market cap, P/E, leverage, profitability), earnings season and sector
Compared five models, from simple to complex: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting and ARIMAX
Trained every model on 2011 to 2018 and tested on 2019 to 2026, data the models never saw, covering COVID, the 2022 rate shock and the AI rally
Results

How well each model forecast next month's volatility (R² | average error in volatility points)

Model	Set A (12 variables)	Set B (all 39 variables)
Linear Regression	0.52 | 8.4	0.27 | 10.9
Ridge Regression	0.52 | 8.4	0.46 | 9.0
Random Forest	0.522 | 8.3	0.528 | 8.5
Gradient Boosting	0.525 | 8.3	0.44 | 9.5
ARIMAX	0.43 | 9.1	0.45 | 9.0

Compared to simple baselines

Forecast	R²	Avg error	Complexity
Last month's volatility (naive)	0.29	10.9	None
VIX × stock's riskiness	0.39	10.6	Very low
Linear Regression (12 variables)	0.52	8.4	Low
Random Forest (all variables)	0.528	8.5	High
Takeaways
Returns are unpredictable. Every model scored at or below 0% R² on unseen data.
Risk is about half predictable. The best models explained about 52% of next month's volatility across 495 stocks.
Simple matched complex. Linear regression on 12 price variables (52.0%) came within 1 point of random forest (52.8%).
More variables can hurt. Adding 27 variables dropped plain linear regression from 52% to 27% when interest rates jumped in 2022 and 2023.
We beat the VIX. By 13 points for individual stocks. For the S&P 500 itself, a simple regression matched or beat the options market's own forecast.
Conclusion

Do investors need complex quant models? No. Quants often do use linear regression, and our data shows why: it was as accurate as machine learning, easier to explain, and harder to break when markets changed.

Retail investors: be skeptical of products that claim to predict returns. No model we tested could.
Financial advisors: use volatility forecasts to set client expectations and size positions.
Risk managers and funds: a simple, transparent model on price history forecasts risk as well as machine learning, and holds up better when the economy shifts.
Limitations
Survivorship bias: only today's S&P 500 members, so companies that failed or left the index are excluded.
One regime dominates the test: 2019 to 2026 was mostly a tech-led bull market.
Monthly data: professionals use intraday data to measure volatility more precisely.
Earnings flag is approximate: based on SEC filing dates, not exact announcement dates.
Data

No data files are needed. The script pulls everything automatically from free public sources:

Yahoo Finance (yfinance): daily stock prices
SEC EDGAR API: company financial statements
FRED (Federal Reserve): VIX and interest rates
Wikipedia: the S&P 500 company list and sectors
Running it
Install the packages:
   pip install yfinance pandas numpy scikit-learn statsmodels requests lxml joblib
Open 502 Project.py and replace the SEC_USER_AGENT placeholder near the top with your own name and email. The SEC requires this to access its data.
Run 502 Project.py.

The first run takes about 15 to 25 minutes while it downloads data; later runs take a few minutes. Results print to the screen and are saved in a results folder.

Files
502 Project.py: the full analysis
502 Project.pdf: our final presentation
