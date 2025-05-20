<a id="readme-top"></a>

<!-- PROJECT LOGO -->

  <h3 align="center">TSLA Price Prediction Model</h3>

  <p align="center">
    A deep learning model to predict Tesla (TSLA) stock price direction using historical technical indicators
    <br />
    <a href="https://github.com/chanz6"><strong>More Personal Projects »</strong></a>
    <br />
  </p>
</div>

[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#project-summary">Project Summary</a>
      <ul>
        <li><a href="#cloning">Cloning</a></li>
        <li><a href="#performance-summary">Performance Summary</a></li>
        <li><a href="#visualizations">Visualizations</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Stock price prediction is a challenging task due to market noise and external volatility. This project explores forecasting Tesla's (TSLA) stock price by combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, enhanced with technical indicators. CNNs capture short-term patterns, while LSTMs model long-term trends in time series data. By integrating momentum, moving averages, and volatility measures, the model aims to improve prediction accuracy and resilience during volatile market periods.

Key stages of the workflow include:

- **Data Collection** – Historical TSLA stock data is retrieved using the Yahoo Finance API via the `yfinance` library.
- **Feature Engineering** – Technical indicators such as EMA, MACD, RSI, ATR, ROC, and Stochastic Oscillators are calculated to capture trend, momentum, and volatility. 
- **Data Processing** – A 30-day sliding window is applied and data is normalized using `MinMaxScaler` to prepare for deep learning input.
- **Model Architecture & Training** – A hybrid CNN-LSTM model is trained using Conv1D and stacked LSTM layers, regularized with dropout and L2 penalties, and optimized with Adam and Huber loss.
- **Evaluation & Metrics** – Model performance is assessed using RMSE, MAE, R², and MAPE, along with visualization of predicted vs actual prices.

Built with Python and Jupyter, this project incorporates libraries like NumPy, Pandas, Matplotlib, Tensorflow, and `ta` for technical analysis. It's a great project for exploring financial time series forecasting, deep learning, and the practical use of technical indicators in predictive modeling.

### Built With

* ![Python][Python]
* ![Pandas][Pandas]
* ![Numpy][Numpy]
* ![Scikit-learn][Sklearn]
* ![TensorFlow][TensorFlow]
* ![Keras][Keras]
* ![Matplotlib][Matplotlib]
* ![YFinance][YFinance]
* ![TA][TA]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Project Summary

### Cloning

This project can be cloned using the following command:

```
git clone https://github.com/chanz6/datasci-TSLA-Price-Forecasting.git
```

### Performance Summary

* _Model training and architecture details can be found in the `price_pred.ipynb` notebook._
* _Visualizations and evaluation plots are included at the end of the notebook for easy interpretation._

The model shows strong performance in capturing general trends in Tesla’s stock price. The CNN-LSTM architecture, supported by technical indicators like EMA, MACD, RSI, and ATR, enables the model to recognize both short and long-term patterns in price movements.

On the test set, the model achieved:
* **R² Score of 0.86**, indicating it explains 86% of the variance in the data
* **Mean absolute percentage error (MAPE) of 8.61%**, demonstrating good average predictive accuracy
* **Root mean squared error (RMSE) of $24.89**, which is reasonable considering TSLA’s high volatility

The model excels at predicting broad trends but occasionally underperforms on sharp, short-term price shifts. These results show the value of combining technical indicators with deep learning, and offer a strong foundation for future improvements such as incorporating external sentiment or macroeconomic features.

### Visualizations

Model Predictions vs Actual Price (Full Timeline):

![Pred vs Actual (Full Timeline)](/images/1.PNG)

TSLA Close Price, Predicted vs Actual (Test Set):

![Pred vs Actual (Test Set)](/images/2.PNG)

Prediction Accuracy, Scatter Plot of Actual vs Predicted:

![Pred Accuracy](/images/3.PNG)

Loss Curve, Training vs Validation

![Loss Curve](/images/4.PNG)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=0077B5
[linkedin-url]: https://www.linkedin.com/in/zachary-chann/
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&logoColor=blue
[Pandas]: https://img.shields.io/badge/Pandas-000bff?style=for-the-badge&logo=pandas&logoColor=purple
[Numpy]: https://img.shields.io/badge/NumPy-ad526f?style=for-the-badge&logo=NumPy&logoColor=blue
[Matplotlib]: https://img.shields.io/badge/Matplotlib-DD0031?style=for-the-badge&logo=matplotlib&logoColor=white
[Yfinance]: https://img.shields.io/badge/yfinance-563D7C?style=for-the-badge&logo=&logoColor=white
[Sklearn]: https://img.shields.io/badge/scikit--learn-FFC0CB?style=for-the-badge&logo=scikitlearn&logoColor=black
[TensorFlow]: https://img.shields.io/badge/tensorflow-orange?style=for-the-badge&logo=tensorflow&logoColor=gold
[Keras]: https://img.shields.io/badge/Keras-yellow?style=for-the-badge&logo=keras&logoColor=gold
[TA]: https://img.shields.io/badge/TA-lightgrey?style=for-the-badge&logo=logoColor=
