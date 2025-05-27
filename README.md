# stock-price-management-202401100300159
# 📈 Stock Price Prediction using Random Forest Regressor

This project builds a regression-based model to **predict next-day stock closing prices** using historical stock market data. It uses the first 7 CSV datasets from a ZIP archive of stock data, applies feature engineering, trains a `RandomForestRegressor`, evaluates its performance using multiple metrics, and visualizes the results.

---

## 📁 Project Structure

📦 stock-price-prediction
├── archive (2).zip # ZIP file containing stock datasets
├── stock_prediction.ipynb # Main notebook (Google Colab / Jupyter compatible)
├── README.md # Project documentation (this file)

yaml
Copy
Edit

---

## 🛠️ Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (RandomForestRegressor)
- Jupyter Notebook / Google Colab

---

## 🚀 How to Run

### In Google Colab:

1. Upload the ZIP file:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload `archive (2).zip`
Update the path in the notebook:

python
Copy
Edit
zip_path = "/content/archive (2).zip"
extract_dir = "stock_data"
Run all cells in stock_prediction.ipynb.

The notebook will:

Extract stock datasets

Use the first 7 CSV files

Train a Random Forest Regressor on each

Predict the next-day closing price

Evaluate and visualize model performance

📊 Evaluation Metrics
For each stock dataset, the following metrics are computed:

Metric	Description
RMSE	Root Mean Squared Error
MAE	Mean Absolute Error
R² Score	Coefficient of determination
Directional Accuracy	% of times predicted direction matched actual
Tolerance Accuracy	% predictions within ±2% of actual value

Each dataset's actual vs. predicted price is plotted along with bar charts for all metrics.

📈 Visualizations
Line plots: Actual vs Predicted Closing Prices

Bar plots: RMSE, MAE, R² Score, Directional Accuracy, and Tolerance Accuracy per stock

🧠 Future Improvements
Use technical indicators (SMA, EMA, RSI, MACD)

Add news sentiment analysis

Test deep learning models (e.g., LSTM, GRU)

Compare with other regressors (e.g., XGBoost, SVR)

📚 Acknowledgements
Data source: [Kaggle / Yahoo Finance]

Libraries: scikit-learn, matplotlib, seaborn, pandas

📬 Contact
For questions or feedback, feel free to open an issue or reach out!

⭐ Star this repo if you found it helpful!
