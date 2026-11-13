from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Used Car Price Predictor</title>
  <style>
    * { box-sizing: border-box; font-family: "Segoe UI", Arial, sans-serif; }
    body {
      margin: 0; min-height: 100vh; display: flex;
      align-items: center; justify-content: center;
      background: linear-gradient(135deg, #1e3c72, #2a5298);
      padding: 20px;
    }
    .card {
      background: #fff; width: 100%; max-width: 520px;
      border-radius: 16px; padding: 32px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    h1 { margin: 0 0 4px; font-size: 24px; color: #1e3c72; }
    .sub { margin: 0 0 24px; color: #666; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    label { display: block; font-size: 13px; font-weight: 600; color: #444; margin-bottom: 6px; }
    input, select {
      width: 100%; padding: 10px 12px; font-size: 15px;
      border: 1px solid #ccc; border-radius: 8px; background: #fafafa;
    }
    input:focus, select:focus { outline: none; border-color: #2a5298; background: #fff; }
    button {
      width: 100%; margin-top: 24px; padding: 14px; font-size: 16px; font-weight: 600;
      color: #fff; background: #2a5298; border: none; border-radius: 8px; cursor: pointer;
    }
    button:hover { background: #1e3c72; }
    .result {
      margin-top: 24px; padding: 20px; text-align: center;
      background: #e8f5e9; border-radius: 12px;
    }
    .result span { display: block; font-size: 13px; color: #2e7d32; }
    .result strong { font-size: 32px; color: #1b5e20; }
    @media (max-width: 480px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="card">
    <h1>Used Car Price Predictor</h1>
    <p class="sub">Enter the car details to get an estimated selling price.</p>
    <form method="post">
      <div class="grid">
        <div><label>New car price (lakh)</label>
          <input name="present_price" type="number" step="0.01" placeholder="e.g. 9.5" required></div>
        <div><label>Kms driven</label>
          <input name="kms" type="number" placeholder="e.g. 40000" required></div>
        <div><label>Car year</label>
          <input name="year" type="number" min="2000" max="2026" placeholder="e.g. 2018" required></div>
        <div><label>Previous owners</label>
          <input name="owner" type="number" min="0" max="3" placeholder="0 to 3" required></div>
        <div><label>Fuel</label>
          <select name="fuel"><option>Petrol</option><option>Diesel</option><option>CNG</option></select></div>
        <div><label>Seller</label>
          <select name="seller"><option>Dealer</option><option>Individual</option></select></div>
        <div style="grid-column: 1 / -1;"><label>Transmission</label>
          <select name="transmission"><option>Manual</option><option>Automatic</option></select></div>
      </div>
      <button type="submit">Predict Price</button>
    </form>
    {% if price %}
    <div class="result">
      <span>Estimated selling price</span>
      <strong>₹{{ price }} lakh</strong>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    if request.method == "POST":
        f = request.form
        car = {
            "Present_Price": float(f["present_price"]),
            "Kms_Driven": int(f["kms"]),
            "Owner": int(f["owner"]),
            "Car_Age": 2026 - int(f["year"]),
            "Fuel_Type_Diesel": int(f["fuel"] == "Diesel"),
            "Fuel_Type_Petrol": int(f["fuel"] == "Petrol"),
            "Seller_Type_Individual": int(f["seller"] == "Individual"),
            "Transmission_Manual": int(f["transmission"] == "Manual"),
        }
        car_df = pd.DataFrame([car], columns=columns)
        price = round(model.predict(car_df)[0], 2)
    return render_template_string(PAGE, price=price)

if __name__ == "__main__":
    app.run(debug=True)