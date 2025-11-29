# 🚗 Car Price Prediction using Machine Learning

## 📌 Description  
A complete end-to-end machine learning project that predicts the selling price of used cars based on features like present price, km driven, fuel type, transmission, ownership, and car age. The project includes data analysis, cleaning, encoding, model training, evaluation, comparison, prediction, and model saving.

---

## 📁 Project Structure
```
├── step1_data_check.py
├── step2_summary_and_featureprep.py
├── step3_clean_encode_split.py
├── step4_linear_regression.py
├── step5_random_forest.py
├── step6_model_comparison.py
├── step7_predict_function.py
├── step8_save_model.py
├── car data.csv
├── car_data_preview.csv
├── car_price_model.pkl
└── model_features.pkl
```

*(Or a single consolidated script if you uploaded all steps together.)*

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Joblib  

---

## 🚀 Project Workflow (Step-by-Step)

### **1️⃣ Data Exploration (Step 1)**
- Loaded CSV file  
- Displayed head, columns, info  
- Checked structure & dataset sanity  

### **2️⃣ Summary & Feature Preparation (Step 2)**
- Identified target column  
- Analyzed missing values  
- Generated summary statistics  
- Created new feature: **Age = CurrentYear – Year**  
- Saved preview file  

### **3️⃣ Data Cleaning & Encoding (Step 3)**
- Removed non-ML columns (`Car_Name`, `Year`)  
- Created target `Selling_Price`  
- One-hot encoded categorical columns:
  - Fuel_Type  
  - Selling_type  
  - Transmission
- Split dataset into train/test  

### **4️⃣ Linear Regression Model (Step 4)**
- Trained Linear Regression  
- Evaluated using R² and MAE  

### **5️⃣ Random Forest Model (Step 5)**
- Trained a stronger RandomForestRegressor  
- Achieved higher accuracy  
- Compared with Linear Regression  

### **6️⃣ Model Comparison (Step 6)**
- Printed side-by-side R² and MAE metrics  
- Random Forest performed best  

### **7️⃣ Price Prediction Function (Step 7)**
- Created a custom function:
  ```python
  predict_price(...)
  ```
- Uses manual one-hot encoding + trained model  

### **8️⃣ Model Saving (Step 8)**
- Saved model as:  
  - `car_price_model.pkl`  
  - `model_features.pkl`  

---

## 📈 Example Prediction Output
```
Predicted Selling Price: 3.85 lakhs
```

---

## 📥 How to Run

### 1️⃣ Install Required Libraries
```bash
pip install pandas numpy scikit-learn joblib
```

### 2️⃣ Predict a New Car Price
Modify the predict function call in Step 7:
```python
predict_price(
    present_price=5.59,
    driven_kms=27000,
    fuel_type="Petrol",
    selling_type="Dealer",
    transmission="Manual",
    owner=0,
    age=11
)
```

---

## 🎯 Features Used
- Present Price  
- Driven Kilometers  
- Owner Count  
- Age  
- Fuel Type (Diesel/Petrol)  
- Seller Type (Dealer/Individual)  
- Transmission (Manual/Automatic)

---

## 📦 Output Files
- **car_price_model.pkl** → Trained Random Forest model  
- **model_features.pkl** → Feature order for prediction  
- **car_data_preview.csv** → Quick cleaned preview  

---

## 🤝 Contributions
Feel free to fork this repository, open issues, or submit improvements.

---

## 📧 Contact
For questions or support, raise an issue in this repository.
