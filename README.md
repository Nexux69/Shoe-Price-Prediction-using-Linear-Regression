# 👟 Shoe Price Prediction App

A beginner-friendly web application that predicts shoe prices based on brand, color, and size using Linear Regression. This tool helps users estimate the price of shoes given their attributes, making it useful for shoppers, sellers, and data enthusiasts interested in price modeling.

---

## 🚀 Live Demo

Check out the deployed app here:  
[🌐 Shoe Price Prediction Streamlit App](https://shoe-price-prediction-faiz-shaikh.streamlit.app/)

---

## 📊 Dataset

- **Source:** [Kaggle - Shoes Price for Various Brands](https://www.kaggle.com/datasets/ashutosh598/shoes-price-for-various-brands)
- **Columns:**
  - `brand`: Brand name of the shoe (e.g., Nike, Adidas)
  - `color`: Color of the shoe (e.g., Black, White)
  - `size`: Shoe size (numerical)
  - `price`: Price in your local currency

---


## 🔄 Project Workflow

### 1. Data Cleaning
- Removed rare colors to focus on the most common ones.
- Outlier removal with Inter-Quartile Range (IQR) to handle abnormal price entries.

### 2. Feature Engineering
- Applied **One-Hot Encoding** to categorical features (`brand`, `color`).

### 3. Model Training
- Trained a **Linear Regression** model to predict prices.
- Saved the trained model using `joblib` for efficient reloading in the app.

---

## 💻 Running Locally

After installing dependencies, start the Streamlit app with:

```bash
streamlit run app.py
```

The app will open in your browser. Input shoe details to get a price prediction!

---

## 📈 Results & Metrics

- **Evaluation Metrics:**  
  - R² Score (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)

- **Sample Plot:**  
  The following plot visualizes Actual vs Predicted prices (you can generate this in the notebook):

  ```python
  import matplotlib.pyplot as plt

  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.xlabel('Actual Price')
  plt.ylabel('Predicted Price')
  plt.title('Actual vs Predicted Shoe Prices')
  plt.show()
  ```

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- Joblib

---

## 🔮 Future Improvements

- Add more features: `discount`, `material`, `brand popularity`, etc.
- Try advanced models: **Random Forest**, **XGBoost** for improved accuracy.
- Deploy with a more interactive UI and support for more shoe attributes.

---

## 👤 Author

**Faiz Shaikh**

---
