# Stock Market Prediction using Support Vector Machine (SVM)

This final year project predicts **hourly stock price movement direction** using Support Vector Machine (SVM) and compares its performance with other models such as Random Forest, XGBoost, and a CNN-LSTM hybrid. The system automatically selects the best-performing model, reuses trained models, and serves predictions using a Flask API with a basic HTML frontend.

---

## 🔧 Features

- Predicts hourly stock movement (up/down)
- Multiple ML models: SVM, Random Forest, XGBoost, Stacking, CNN-LSTM
- Model selection based on highest accuracy
- Real-time Yahoo Finance data support
- Flask-based API with HTML frontend
- Saves and reuses trained models (`.pkl` / `.h5`)
- Scaler persistence (`scaler.pkl`)
- Uses NIFTY50 dataset

---

## 📁 Project Structure

```
src/
├── model/                         # Saved models and scaler
│   ├── cnn_lstm_model.h5
│   ├── randomforest_model.pkl
│   ├── scaler.pkl
│   ├── stacking_model.pkl
│   ├── svm_model.pkl
│   └── xgboost_model.pkl
│
├── templates/                     # HTML template for front-end
│   └── index.html
│
├── training/                      # Model training script
│   └── training.py
│
├── app.py                         # Flask app to serve predictions
├── NIFTY50_all.csv                # Dataset
```

Other files:

```
.venv/                              # Python virtual environment
.gitignore
NIFTY50_all.csv                    # Dataset copy (root level)
README.md
requirements.txt
sample.py                          # Optional script for testing
stock_market_prediction.zip        # Compressed project folder
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-market-prediction.git
cd stock-market-prediction
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Models

Navigate to the `src/training/` folder and run:

```bash
python training.py
```

This trains models and saves them into the `src/model/` directory.

### 5. Start the Flask App

From the `src/` folder:

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` to access the web interface.

---

## 🧠 Models Used

- **Support Vector Machine (SVM)**
- **Random Forest**
- **XGBoost**
- **Stacking Classifier**
- **CNN-LSTM (Deep Learning)**

Model selection is done dynamically based on their prediction accuracy.

---

## 📊 Dataset

- Dataset: NIFTY50 historical stock data
- Format: CSV
- Columns: Date, Open, High, Low, Close, Volume, etc.

---

## ⚙️ Notes

- Training was done using **CPU only** (no GPU)
- Models are saved in `src/model/` for reuse
- Uses `scaler.pkl` for consistent data scaling
- Real-time prediction logic can be added via Yahoo Finance integration

---


## 👨‍💻 Author

**Paramesh Kumar Selvaraj** – Final Year Engineering Student

**Purusothaman Rajan** – Final Year Engineering Student

**Tharun Murugavel** – Final Year Engineering Student  

For queries, contact via LinkedIn or email.