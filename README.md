# New-Valuation-Measure-Replication-With-Annual-Volatility
This replication of my work arXiv:1905.04603 A New Stock Market Valuation Measure with Applications to Retirement Planning with Annual Volatility done by Angel Piotrowski. See https://arxiv.org/abs/1905.04603 for the original work. 

I explained in brief this previous work in the blog post https://my-finance.org/2025/02/20/replicated-arxiv1905-04603-with-annual-volatility/ Current work is explained in another post https://my-finance.org/2025/03/24/new-valuation-measure-replicated-volatility/ which continues yet another post https://my-finance.org/2025/02/20/new-bubble-measure-replicated-with-stochastic-volatility/

replicate.py is the replication of my results with CPI data for December of each year rather than the January of the following year, and for S&P 500 data from the last trading day of the year, rather than January daily average of the following year. norm-idy.py is the version of this analysis with division of total returns minus earnnings growth.

Added later: new-measure.py is the simulator version of this, allowing you to choose the averaging window between 1 and 10 years, and nominal vs real. We do not have volatility factor in the main regression. But this fsactor is statistically significant, with p < 1% from the Student T-test. https://my-finance.org/2025/03/26/annual-simulator-with-volatility-and-the-new-valuation-measure-of-sp-500/

Added later: compare-bubble-logyield.py shows the bubble (the new valuation measure) is better than the log yield as returns predictor. https://my-finance.org/2025/03/29/using-both-new-valuation-measure-and-cape/
