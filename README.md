# Wine Type Classification using Neural Networks

This project builds a machine learning model to classify **red vs. white wine** using chemical properties from the Wine Quality dataset. The model uses a **Keras Sequential neural network** to learn patterns and predict wine type based on features like acidity, alcohol content, density, etc.

---

## 📂 Dataset

### **Source**

* Red Wine: [https://media.geeksforgeeks.org/wp-content/uploads/20251001112542021865/redwinequality.csv]
* White Wine: [https://media.geeksforgeeks.org/wp-content/uploads/20251001112608870752/whitewinequality.csv]
### **Features**

* Dataset includes 12 chemical measurements.
* Added **type** column:

  * `1` → Red Wine
  * `0` → White Wine

### **Combined Dataset Shape**

* ~6500+ rows
* 13 columns (including target)

---

## 🧠 Model Overview

A fully connected neural network (Dense layers) was used.

### **Model Architecture**

```
Input Layer: 12 features
Hidden Layer 1: Dense(12), activation='relu'
Hidden Layer 2: Dense(9), activation='relu'
Output Layer: Dense(1), activation='sigmoid'
```

### **Compilation**

```
loss='binary_crossentropy'
optimizer='adam'
metrics=['accuracy']
```

---

## 📊 Training

```
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)
```

* Train/Test split: 80/20
* `random_state=45` ensures reproducibility

---

## 🧪 Testing the Model

### **Evaluate**

```
loss, accuracy = model.evaluate(X_test, y_test)
```

### **Predict**

```
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
```

---

## 📈 Visualizations

The notebook includes:

* Alcohol distribution histograms
* Feature comparisons
* Confusion matrix
* Classification report

---

## 📁 Project Structure

```
📦 wine-type-classification
│
├── wine_type_prediction.ipynb   # Main notebook
├── README.md                    # Documentation

```

---

## 🚀 How to Run This Project

### **2️⃣ Create a virtual environment**

```
python -m venv venv
```

Activate:

* Windows: `venv\\Scripts\\activate`
* Mac/Linux: `source venv/bin/activate`

### **3️⃣ Install dependencies**

```
pip install -r requirements.txt
```

### **4️⃣ Run the notebook**

```
jupyter notebook
```

---

## 📌 Key Takeaways

* Model successfully classifies red vs white wine.
* Neural networks learn non-linear relationships via Dense + ReLU layers.
* Dataset imbalance and feature scaling can affect accuracy.
* Demonstrates binary classification using Keras.

---

## 🧪 Future Improvements

* Add data normalization
* Experiment with deeper architectures
* Try other activation functions
* Compare with classical ML models (RF, SVM)
* Perform k-fold cross-validation


