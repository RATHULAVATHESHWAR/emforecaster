# ⚡ EMF Exposure Prediction System

A web-based application to predict Electromagnetic Field (EMF) exposure using user inputs such as frequency, power, and distance. The system visualizes exposure levels and classifies safety risks.

---

## 🚀 Features

- EMF prediction using inverse-square model
- User input (Frequency, Power, Distance)
- Graph visualization (EMF vs Distance)
- Safety classification (Safe / Moderate / High Risk)
- Web interface using Flask
- Deployed on Ubuntu cloud server

---

## 🧠 Model Explanation

EMF = (Power × Frequency Factor) / Distance²

Where:
- Frequency Factor = Frequency / 1000
- Distance follows inverse-square law

---

## 🖥️ Tech Stack

- Python 3
- Flask
- NumPy
- Matplotlib
- HTML + CSS

---

## ⚙️ Installation & Setup (Ubuntu Server)

### 1. Update system
sudo apt update && sudo apt upgrade -y

### 2. Install Python & pip
sudo apt install python3 python3-pip python3-venv -y

### 3. Clone project (or upload files)
git clone <your-repo-link>
cd emforecaster

### 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

### 5. Install dependencies
pip install flask numpy matplotlib

---

## ▶️ Run Application

cd ~/emforecaster
source venv/bin/activate
python app.py

---

## 🌐 Access in Browser

http://YOUR-SERVER-IP:5000

---

## 📊 Usage

1. Enter:
   - Frequency (Hz)
   - Power (W)
   - Distance (m)

2. Click "Predict"

3. View:
   - EMF Exposure value
   - Risk level
   - Graph visualization

---

## 📈 Output

- EMF exposure value
- Risk classification
- Graph showing EMF vs Distance
- Safety zones visualization

---

## 🎓 Project Explanation

This system predicts EMF exposure based on physical parameters using a normalized inverse-square model. It provides both numerical and graphical outputs, making it useful for analyzing electromagnetic safety levels.

---

## 🚀 Future Improvements

- Integrate AI-based forecasting model
- Export results as PDF
- Interactive graphs
- Deploy with domain

---

## 👨‍💻 Author

Final Year Project  
EMF Exposure Prediction System
