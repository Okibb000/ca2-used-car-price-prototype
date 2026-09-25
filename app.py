import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


FEATURES = ["Brand", "Manufacture_Year", "Mileage_km", "Horsepower_PS", "Seats"]
RANDOM_STATE = 42


@st.cache_resource
def load_and_train():
    df = pd.read_csv("Lab04_hk_car_price.csv")
    X_all = pd.get_dummies(
        df[FEATURES].copy(), columns=["Brand"], drop_first=True, dtype=float
    )
    y_all = df["Price_HKD"]
    X_train, _, y_train, _ = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return df, model, list(X_all.columns)


df, model, model_columns = load_and_train()

st.title("HK Used Car Price Estimator")
st.caption("HO Sin Man 250596968 CA2 Prototype")
st.write(
    "Estimate a Hong Kong used car resale price from Motor City transaction data. "
    "This is a quote ballpark, not an official valuation."
)

brand = st.selectbox("Brand", sorted(df["Brand"].dropna().unique().tolist()))
year = st.slider(
    "Manufacture year",
    int(df["Manufacture_Year"].min()),
    int(df["Manufacture_Year"].max()),
    int(df["Manufacture_Year"].median()),
)
mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=int(df["Mileage_km"].max()),
    value=int(df["Mileage_km"].median()),
    step=1000,
)
horsepower = st.slider(
    "Horsepower (PS)",
    int(df["Horsepower_PS"].min()),
    int(df["Horsepower_PS"].max()),
    int(df["Horsepower_PS"].median()),
)
seats = st.slider(
    "Seats",
    int(df["Seats"].min()),
    int(df["Seats"].max()),
    int(df["Seats"].median()),
)

if st.button("Estimate Price", type="primary"):
    row = pd.DataFrame(
        [
            {
                "Brand": brand,
                "Manufacture_Year": year,
                "Mileage_km": mileage,
                "Horsepower_PS": horsepower,
                "Seats": seats,
            }
        ]
    )
    row = pd.get_dummies(row, columns=["Brand"], drop_first=False, dtype=float)
    row = row.reindex(columns=model_columns, fill_value=0)
    price = max(0, model.predict(row)[0])
    st.success(f"Estimated price: HK${price:,.0f}")
    st.caption(
        "Use this estimate as a talking range. Check the vehicle condition and current "
        "market before giving a final quote."
    )
