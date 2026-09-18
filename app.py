from flask import Flask, render_template, request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Cleaned_Car_data.csv")

# Remove unwanted index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

X = df.drop(columns=["Price"])
y = df["Price"]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train on complete dataset
model.fit(X, y)

companies = sorted(df["company"].dropna().astype(str).unique())
fuel_types = sorted(df["fuel_type"].dropna().astype(str).unique())
car_names = sorted(df["name"].dropna().astype(str).unique())


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    form_data = {
        "name": "",
        "company": "",
        "year": "",
        "kms_driven": "",
        "fuel_type": ""
    }

    if request.method == "POST":
        form_data = {
            "name": request.form.get("name", "").strip(),
            "company": request.form.get("company", "").strip(),
            "year": request.form.get("year", "").strip(),
            "kms_driven": request.form.get("kms_driven", "").strip(),
            "fuel_type": request.form.get("fuel_type", "").strip()
        }

        try:
            year = int(form_data["year"])
            kms_driven = float(form_data["kms_driven"])

            if not all([form_data["name"], form_data["company"], form_data["fuel_type"]]):
                raise ValueError("Please fill in all fields.")

            if year < 1900 or year > 2100:
                raise ValueError("Please enter a valid year.")

            if kms_driven < 0:
                raise ValueError("Kilometers cannot be negative.")

            new_car = pd.DataFrame({
                "name": [form_data["name"]],
                "company": [form_data["company"]],
                "year": [year],
                "kms_driven": [kms_driven],
                "fuel_type": [form_data["fuel_type"]]
            })

            prediction = max(0, float(model.predict(new_car)[0]))

        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Unable to predict. Please check your input."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        companies=companies,
        fuel_types=fuel_types,
        car_names=car_names
    )


if __name__ == "__main__":
    app.run(debug=True)
