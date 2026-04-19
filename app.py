from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # =========================
        # 1. GET INPUTS
        # =========================
        freq = float(request.form['frequency'])
        power = float(request.form['power'])
        distance = float(request.form['distance'])

        # =========================
        # 2. EMF MODEL
        # =========================
        freq_factor = freq / 1000
        emf = (power * freq_factor) / (distance**2 + 1e-6)

        # =========================
        # 3. SAFETY LEVEL
        # =========================
        if emf < 1:
            level = "Safe"
            color = "green"
        elif emf < 5:
            level = "Moderate"
            color = "orange"
        else:
            level = "High Risk"
            color = "red"

        # =========================
        # 4. GRAPH DATA
        # =========================
        distances = np.linspace(1, distance + 10, 100)
        emf_values = (power * freq_factor) / (distances**2)

        # =========================
        # 5. GRAPH
        # =========================
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(10, 6))

        plt.plot(distances, emf_values, color='blue', linewidth=3, label="EMF Curve")
        plt.scatter(distance, emf, color='red', s=120, label="Your Input")

        # Safety zones
        plt.axhspan(0, 1, color='green', alpha=0.1)
        plt.axhspan(1, 5, color='orange', alpha=0.1)
        plt.axhspan(5, max(emf_values)*1.1, color='red', alpha=0.1)

        # Safety lines
        plt.axhline(y=1, color='green', linestyle='--')
        plt.axhline(y=5, color='red', linestyle='--')

        plt.xlabel("Distance (m)")
        plt.ylabel("EMF Exposure")
        plt.title("EMF Exposure vs Distance")
        plt.grid(True)

        plt.legend()
        plt.tight_layout()

        # =========================
        # 6. SAVE GRAPH
        # =========================
        if not os.path.exists("static"):
            os.makedirs("static")

        plt.savefig("static/output.png", dpi=300)
        plt.close()

        # =========================
        # 7. RETURN RESULT PAGE
        # =========================
        return render_template(
            "result.html",
            freq=freq,
            power=power,
            distance=distance,
            emf=round(emf, 4),
            level=level,
            color=color
        )

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
