# ⚡ EMF Exposure Prediction System

A web-based application to predict Electromagnetic Field (EMF) exposure using user inputs such as frequency, power, and distance.

---

## 🚀 Features

- EMF prediction using inverse-square model
- User input (Frequency, Power, Distance)
- Graph visualization
- Safety classification (Safe / Moderate / High Risk)
- Flask web application
- Deployed on Ubuntu server

---

## 🧠 Model Explanation

EMF = (Power × Frequency Factor) / Distance²

Where:
- Frequency Factor = Frequency / 1000
- EMF decreases with distance (inverse-square law)

---

## 🖥️ Tech Stack

- Python 3
- Flask
- NumPy
- Matplotlib
- HTML + CSS

---

## ⚙️ Installation & Setup (Ubuntu)

### 1. Update system
sudo apt update && sudo apt upgrade -y

---

### 2. Install Python
sudo apt install python3 python3-pip python3-venv -y

---

### 3. Go to project folder
cd ~/emforecaster

---

### 4. Create virtual environment
python3 -m venv venv

---

### 5. Activate environment
source venv/bin/activate

---

### 6. Install dependencies
pip install flask numpy matplotlib

---

## ▶️ Run Web Application

### Step 1: Go to project folder
cd ~/emforecaster

### Step 2: Activate environment
source venv/bin/activate

### Step 3: Run Flask app
python app.py

---

## 🌐 Access Application

Open browser and go to:

http://YOUR-SERVER-IP:5000

Example:
http://172.31.37.211:5000

---

## 📊 Usage

1. Enter:
   - Frequency (Hz)
   - Power (W)
   - Distance (m)

2. Click **Predict**

3. View:
   - EMF Exposure value
   - Risk level
   - Graph visualization

---

## 📈 Output

- EMF exposure value
- Risk classification
- EMF vs Distance graph
- Safety zones visualization

---

## 🎓 Project Explanation

This system predicts EMF exposure using a normalized inverse-square model. It classifies risk levels and provides graphical visualization for better understanding.

---

## 🚀 Future Improvements

- Add AI-based prediction model
- Export results as PDF
- Interactive graphs
- Deploy with domain

---

## 👨‍💻 Author

Final Year Project  
EMF Exposure Prediction System
