from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

df = None
model = None
le_country = None
ready = False
message = ""

# =========================
# COLUMN AUTO-DETECTION
# =========================
def find_column(cols, keys):
    for c in cols:
        for k in keys:
            if k in c:
                return c
    return None

# =========================
# DATA PREPARATION
# =========================
def prepare_model(raw):
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "")

    col_date = find_column(raw.columns, ["date"])
    col_qty = find_column(raw.columns, ["qty", "quantity"])
    col_price = find_column(raw.columns, ["price", "unitprice", "amount"])
    col_country = find_column(raw.columns, ["country", "region"])
    col_product = find_column(raw.columns, ["description", "product", "item"])
    col_category = find_column(raw.columns, ["category", "type"])

    missing = [n for n, c in {
        "Date": col_date,
        "Quantity": col_qty,
        "Price": col_price,
        "Country": col_country,
        "Product": col_product
    }.items() if not c]

    if missing:
        raise ValueError(f"Dataset missing columns: {', '.join(missing)}")

    data = raw.copy()
    data["date"] = pd.to_datetime(data[col_date], errors="coerce")
    data["quantity"] = pd.to_numeric(data[col_qty], errors="coerce")
    data["price"] = pd.to_numeric(data[col_price], errors="coerce")
    data["country"] = data[col_country].astype(str)
    data["product"] = data[col_product].astype(str)

    if col_category:
        data["category"] = data[col_category].astype(str)
    else:
        data["category"] = "General"

    data.dropna(inplace=True)
    data = data[(data["quantity"] > 0) & (data["price"] > 0)]

    data["month"] = data["date"].dt.month
    data["revenue"] = data["quantity"] * data["price"]

    data["demand"] = np.where(data["quantity"] >= data["quantity"].median(), 1, 0)

    le = LabelEncoder()
    data["country_enc"] = le.fit_transform(data["country"])

    X = data[["month", "price", "country_enc"]]
    y = data["demand"]

    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, y)

    return data, clf, le

# =========================
# RECOMMENDATIONS
# =========================
def get_recommendations(month, price, country, top_n):
    temp = df.copy()

    try:
        enc = le_country.transform([country])[0]
    except:
        enc = temp["country_enc"].mode()[0]

    probs = model.predict_proba(
        temp[["month", "price", "country_enc"]]
    )[:, 1]

    temp["demand_prob"] = (probs * 100).round(2)

    temp = temp[
        (temp["month"] == month) &
        (temp["country_enc"] == enc)
    ]

    temp["price_diff"] = abs(temp["price"] - price)

    return (
        temp.sort_values(["demand_prob", "revenue"], ascending=False)
        .groupby("product")
        .agg({
            "quantity": "sum",
            "revenue": "sum",
            "demand_prob": "mean"
        })
        .sort_values("demand_prob", ascending=False)
        .head(top_n)
        .reset_index()
        .to_dict(orient="records")
    )

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    global df, model, le_country, ready, message

    recs = []

    if request.method == "POST":
        if "file" in request.files:
            f = request.files["file"]
            path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(path)

            raw = pd.read_csv(path, encoding="ISO-8859-1")
            df, model, le_country = prepare_model(raw)
            ready = True
            message = "Dataset loaded successfully ✅"

        elif ready:
            recs = get_recommendations(
                int(request.form["month"]),
                float(request.form["price"]),
                request.form["country"],
                int(request.form["top_n"])
            )

    countries = sorted(df["country"].unique()) if ready else []

    top10 = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .to_dict(orient="records")
    ) if ready else []

    category_chart = (
        df.groupby("category")["revenue"]
        .sum()
        .reset_index()
        .to_dict(orient="records")
    ) if ready else []

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>Smart Recommendation System</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{font-family:Arial;background:#f4f6f9;padding:40px}
.box{background:white;padding:30px;width:900px;margin:auto;border-radius:12px}
input,select,button{width:100%;padding:10px;margin-top:10px}
table{width:100%;margin-top:20px;border-collapse:collapse}
th,td{border:1px solid #ccc;padding:10px;text-align:center}
button{background:#2563eb;color:white;border:none}
</style>
</head>
<body>

<div class="box">
<h2>📦 Smart Product Recommendation System</h2>

<form method="POST" enctype="multipart/form-data">
<input type="file" name="file" required>
<button>Upload Dataset</button>
</form>

{% if ready %}
<hr>

<form method="POST">
<input name="month" type="number" min="1" max="12" placeholder="Month" required>
<input name="price" type="number" step="0.01" placeholder="Target Price" required>
<select name="country">
{% for c in countries %}<option>{{c}}</option>{% endfor %}
</select>
<input name="top_n" type="number" value="5">
<button>Get Recommendations</button>
</form>

<h3>📈 Demand Probability (%)</h3>
<table>
<tr><th>Product</th><th>Probability %</th><th>Revenue</th></tr>
{% for r in recs %}
<tr>
<td>{{r.product}}</td>
<td>{{r.demand_prob}}</td>
<td>{{r.revenue|round(2)}}</td>
</tr>
{% endfor %}
</table>

<h3>🔝 Top 10 Products</h3>
<table>
<tr><th>Product</th><th>Revenue</th></tr>
{% for t in top10 %}
<tr><td>{{t.product}}</td><td>{{t.revenue|round(2)}}</td></tr>
{% endfor %}
</table>

<h3>📊 Category Revenue Chart</h3>
<canvas id="catChart"></canvas>

<script>
const ctx=document.getElementById("catChart");
new Chart(ctx,{
 type:"bar",
 data:{
  labels: {{category_chart | map(attribute='category') | list | tojson}},
  datasets:[{
   data: {{category_chart | map(attribute='revenue') | list | tojson}}
  }]
 }
});
</script>

{% endif %}
<p>{{message}}</p>
</div>
</body>
</html>
""", ready=ready, countries=countries,
       recs=recs, top10=top10,
       category_chart=category_chart,
       message=message)

if __name__ == "__main__":
    app.run(debug=True)
