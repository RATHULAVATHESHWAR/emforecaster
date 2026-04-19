# ⚡ EMF Exposure Prediction System

A web-based application to predict Electromagnetic Field (EMF) exposure using Frequency, Power, and Distance. The system provides risk classification and graphical visualization.

---

## 🎯 Objective

To estimate EMF exposure levels and classify them into safety categories using a physics-based inverse-square model, presented through a web interface.

---

## 🧠 Model Used

EMF = (Power × Frequency Factor) / Distance²

Where:

* Frequency Factor = Frequency / 1000
* Based on inverse-square law (EMF decreases with distance)

---

## 🛠️ Tools & Technologies Used

| Tool          | Purpose                             |
| ------------- | ----------------------------------- |
| Python 3      | Core programming language           |
| Flask         | Web framework for backend           |
| NumPy         | Mathematical calculations           |
| Matplotlib    | Graph visualization                 |
| HTML/CSS      | Frontend UI                         |
| Ubuntu Server | Deployment environment              |
| Git & GitHub  | Version control and project hosting |

---

## ⚙️ Installation & Setup (Ubuntu)

### Step 1: Update system

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Python and tools

```bash
sudo apt install python3 python3-pip python3-venv -y
```

---

## 📁 Project Setup

### Step 3: Navigate to project folder

```bash
cd ~/emforecaster
```

### Step 4: Create virtual environment

```bash
python3 -m venv venv
```

### Step 5: Activate environment

```bash
source venv/bin/activate
```

### Step 6: Install dependencies

```bash
pip install flask numpy matplotlib
```

---

## ▶️ Run the Web Application

### Step 1: Go to project directory

```bash
cd ~/emforecaster
```

### Step 2: Activate environment

```bash
source venv/bin/activate
```

### Step 3: Run the application

```bash
python app.py
```

---

## 🌐 Access the Application

Open your browser:

```bash
http://YOUR-SERVER-IP:5000
```

Example:

```bash
http://172.31.37.211:5000
```

---

## 📊 How to Use

1. Enter:

   * Frequency (Hz)
   * Power (W)
   * Distance (m)

2. Click **Predict**

3. The system will display:

   * EMF exposure value
   * Risk level (Safe / Moderate / High Risk)
   * Graph visualization

---

## 📈 Output Description

* Numerical EMF value
* Safety classification
* Graph showing EMF vs Distance
* Highlighted input point
* Safety zones:

  * 🟢 Green (Safe)
  * 🟠 Orange (Moderate)
  * 🔴 Red (High Risk)

---

## 📌 Notes

* Ensure the virtual environment is activated before running the app.
* Modify `app.py` if you want to change thresholds or visualization style.
* Suitable for educational and demonstration purposes.

---

## 🚀 Future Improvements

* Add real-time sensor integration
* Deploy using Docker
* Add user authentication
* Improve UI with modern frameworks

---

## 👨‍💻 Author

Developed as a physics-based EMF prediction and visualization system.
