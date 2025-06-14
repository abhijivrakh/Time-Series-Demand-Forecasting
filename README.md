# 📈 Time Series Demand Forecasting

A complete end-to-end machine learning project that forecasts demand based on time series sales data. This project includes data preprocessing, feature engineering, model training using XGBoost, Streamlit UI, and deployment on the cloud.

🚀 **Live Demo**: [Click to Open App](https://time-series-demand-forecasting-ybcexp64t2lw7wxlqlzc8z.streamlit.app/)

---


## 🔧 Project Structure

```
.
├── data/                    # Raw and processed datasets
├── models/                 # Trained model (.pkl)
├── src/                    # Python scripts (feature engineering, training)
├── Streamlit.py            # Streamlit UI
├── requirements.txt        # Python dependencies
├── Dockerfile              # (Optional) Docker setup
└── README.md
```

---

## 💻 Features

* Feature engineering with lag/rolling statistics
* XGBoost model with GridSearchCV
* Streamlit UI to upload new data and get predictions
* Downloadable prediction CSV
* Cloud deployment on Streamlit Community Cloud

---

## ⚙️ CI/CD (Optional Setup)

If using GitHub Actions:

1. Create `.github/workflows/deploy.yml`
2. Add the following sample workflow:

```yaml
name: Deploy Streamlit App

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Lint and test
      run: |
        echo "✅ Add unit tests here if needed."

    - name: Deploy (Manual Step Required)
      run: |
        echo "📦 Deploy manually on Streamlit Cloud via GitHub repo."
```

> Note: Streamlit Cloud currently requires manual deployment via linking GitHub repository.

---

## 🤝 Contribution Guidelines

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request

---

## 🙏 Credits

This project is built with:

* [Streamlit](https://streamlit.io/)
* [XGBoost](https://xgboost.ai/)
* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [joblib](https://joblib.readthedocs.io/en/latest/)

Developed and maintained by [@abhijivrakh](https://github.com/abhijivrakh)

---

## 📜 License

This project is licensed under the MIT License.
