# CreditMomentum
Momentum Strategies Applied in the Credit Market

## Credit Default Swap Indices Kalman Filter
Due to Bloomberg Terminal Data options there were only two indices that had CDS OAS and CDS Index
1. CDX EM CDSI GEN 5y
2. CDX HY CDSI GEN 5y

The log spreads 
![image](https://github.com/user-attachments/assets/3313f801-0f83-4f9f-9e92-4c1cfa1f2aa2)

Then fit the data via Kalman Filter to get the smoothed component and the mean-reverting component
![image](https://github.com/user-attachments/assets/56769348-e533-4c25-b72e-226bb051d2e7)

Using a trend following method based 1m and 3m betas of the smoothed component and then picking the signal that has the maximum lagged 1m roling sharpe provides
![image](https://github.com/user-attachments/assets/3a323077-c60f-43fe-ae39-dc42020bde9b)

The signals outperform on a relative-sharpe standpoint
![image](https://github.com/user-attachments/assets/9781fc36-9578-48b4-9a02-712558226356)

A Z-score can be applied to the mean-reverting component to find richness and cheapness
![image](https://github.com/user-attachments/assets/7a8cb5c5-6acd-4f5c-9a44-ea11969c6f94)

Analyzing all of the strategies together provides a considerably low correlation. Although it should be noted that the Kalman Filter isn't predicting out-of-sample

![image](https://github.com/user-attachments/assets/5a1b5dc7-5ed5-4604-8d59-c8d3d4187903)
