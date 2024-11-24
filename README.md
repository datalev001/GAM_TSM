# Generalized Additive Nonlinear Models for Time Series Forecasting
A Novel Framework Integrating Bounded Nonlinear Functions into ARMA Models for Enhanced Forecasting Accuracy and Robustness with Sparse Data
Time series forecasting plays an important role in real-world applications like finance, economics, and environmental studies. Traditional models like ARMA are popular because they're simple and easy to understand. But ARMA models assume everything is linear, which makes it hard for them to capture the messy, nonlinear patterns often seen in real data. On top of that, sparse time-series data can make advanced models like neural networks struggle since they need a lot of data and parameters to work well.
In this paper, I propose a novel approach called the GAM Time Series Forecasting Model. This method combines bounded nonlinear functions into the traditional ARMA framework, replacing each autoregressive (AR) term with a function f(x∣a,b), where a and b are unified parameters. The proposed model leverages the bounded nature of f(x∣a,b) to ensure stability and regularity while preserving the interpretability and flexibility of the traditional ARMA model.
The primary contributions of this paper are:
A mathematical framework for incorporating bounded nonlinear functions in time-series forecasting.
The development of a loss function and optimization procedure for estimating the parameters a and b in f(x∣ a,b).
Empirical evidence supporting the robustness of this method for sparse data and its advantages over traditional methods.
