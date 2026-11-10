
import os
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="APL Logistics | Late Delivery Risk",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, DATA_FILE = os.path.join(BASE_DIR, "APL_Logistics.csv.gz")

MODEL_FILE = os.path.join(BASE_DIR, "late_delivery_model.joblib")
PREDICTIONS_FILE = os.path.join(
    BASE_DIR, "late_delivery_risk_predictions.csv"
)

TARGET = "Late_delivery_risk"


@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(DATA_FILE, encoding="latin1")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def get_model_details(bundle):
    if isinstance(bundle, dict):
        model = bundle.get("model")
        features = bundle.get("feature_columns", [])
        threshold = float(bundle.get("threshold", 0.5))
    else:
        model = bundle
        features = []
        threshold = 0.5

    if model is None:
        raise ValueError("No trained model was found in the saved file.")

    # Prefer the exact input features recorded by the fitted pipeline.
    fitted_features = getattr(model, "feature_names_in_", None)

    if fitted_features is not None:
        features = list(fitted_features)
    else:
        features = list(features)

        # If feature names are unavailable, check the fitted input count.
        expected_count = getattr(model, "n_features_in_", None)

        if expected_count is not None and len(features) != expected_count:
            raise ValueError(
                "The saved model's expected feature count does not match "
                "the feature list in the model bundle. Please check the "
                "original training notebook before making predictions."
            )

    if not features:
        raise ValueError("The model's input feature list is missing.")

    return model, features, threshold


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def prepare_model_input(frame, features, reference_df):
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(
            "Missing model input columns: " + ", ".join(missing)
        )

    result = frame[features].copy()

    for col in features:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            result[col] = pd.to_numeric(result[col], errors="coerce")
            result[col] = result[col].fillna(
                pd.to_numeric(reference_df[col], errors="coerce").median()
            )
            if result[col].isna().any():
                result[col] = result[col].fillna(0)
        else:
            result[col] = result[col].where(
                result[col].notna(), "Unknown"
            ).astype(str)

    return result


def late_rate_table(frame, group_col):
    result = frame.groupby(group_col, dropna=False)[TARGET].agg(
        Orders="count", Late_Orders="sum", Late_Rate="mean"
    ).reset_index()

    result["Late_Rate"] = result["Late_Rate"] * 100
    return result.sort_values("Late_Rate", ascending=False)


st.title("📦 APL Logistics")
st.subheader("Machine Learning-Based Late Delivery Risk Prediction")
st.caption(
    "Global Supply Chain Operations | Predictive Analytics "
    "and Decision Support Dashboard"
)

try:
    df = load_data()
    bundle = load_model()
    model, feature_columns, threshold = get_model_details(bundle)

    if TARGET not in df.columns:
        st.error(f"Target column '{TARGET}' is missing from the dataset.")
        st.stop()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()

except Exception as e:
    st.error(f"Project files could not be loaded: {e}")
    st.info(
        "Confirm that APL_Logistics.csv and "
        "late_delivery_model.joblib are beside app.py."
    )
    st.stop()


total_orders = len(df)
late_count = int((df[TARGET] == 1).sum())
on_time_count = int((df[TARGET] == 0).sum())
late_rate = late_count / total_orders * 100 if total_orders else 0

st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio(
    "Choose a module",
    [
        "Delay Risk Overview",
        "Order Risk Prediction",
        "Region & Shipping Analysis",
        "Operations Action Panel"
    ]
)

st.sidebar.divider()
st.sidebar.caption(f"Dataset: {total_orders:,} orders")
st.sidebar.caption(f"Model threshold: {threshold:.2f}")


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------
if page == "Delay Risk Overview":
    st.header("Delivery Performance Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{total_orders:,}")
    c2.metric("Late Orders", f"{late_count:,}")
    c3.metric("On-Time Orders", f"{on_time_count:,}")
    c4.metric("Late Delivery Rate", f"{late_rate:.2f}%")

    st.markdown("### Delivery Status Distribution")
    status_data = pd.DataFrame({
        "Delivery Status": ["Late", "On Time"],
        "Orders": [late_count, on_time_count]
    }).set_index("Delivery Status")
    st.bar_chart(status_data)

    st.markdown("### Key Business Findings")
    a, b = st.columns(2)

    with a:
        st.info(
            f"{late_count:,} orders are labelled late in the dataset. "
            "These records can help identify recurring operational risks."
        )

    with b:
        st.info(
            f"The observed late-delivery rate is {late_rate:.2f}%. "
            "Use this as a historical baseline, not a future guarantee."
        )

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button(
        "Download Dataset Preview",
        data=df.head(1000).to_csv(index=False).encode("utf-8"),
        file_name="apl_logistics_dataset_preview.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# PAGE 2: SINGLE AND BATCH PREDICTIONS
# --------------------------------------------------
elif page == "Order Risk Prediction":
    st.header("Order Risk Prediction")
    st.write(
        "Estimate late-delivery risk for a single order or for a "
        "batch of orders using the saved machine learning model."
    )

    missing_features = [
        c for c in feature_columns if c not in df.columns
    ]

    if missing_features:
        st.error(
            "Model input columns are missing from the dataset: "
            + ", ".join(missing_features)
        )
        st.stop()

    mode = st.radio(
        "Prediction mode",
        ["Single Order", "Batch Predictions"],
        horizontal=True
    )

    if mode == "Single Order":
        values = {}

        with st.form("single_order_form"):
            st.markdown("### Enter Order Details")

            left, right = st.columns(2)

            for i, col in enumerate(feature_columns):
                container = left if i % 2 == 0 else right

                with container:
                    series = df[col]

                    if pd.api.types.is_numeric_dtype(series):
                        clean = pd.to_numeric(series, errors="coerce")
                        default = clean.median()
                        if pd.isna(default):
                            default = 0

                        values[col] = st.number_input(
                            col,
                            value=float(default),
                            key=f"single_{col}"
                        )
                    else:
                        options = (
                            series.dropna().astype(str).unique().tolist()
                        )
                        options = sorted(options)

                        if not options:
                            options = ["Unknown"]

                        values[col] = st.selectbox(
                            col, options, key=f"single_{col}"
                        )

            submitted = st.form_submit_button(
                "Predict Late-Delivery Risk",
                use_container_width=True
            )

        if submitted:
            try:
                input_df = pd.DataFrame([values])
                input_df = prepare_model_input(
                    input_df, feature_columns, df
                )

                probability = float(model.predict_proba(input_df)[0][1])
                high_risk = probability >= threshold

                st.markdown("### Prediction Result")
                st.metric(
                    "Estimated Late-Delivery Probability",
                    f"{probability:.1%}"
                )
                st.progress(min(max(probability, 0.0), 1.0))
                st.caption(
                    f"Decision threshold: {threshold:.2f}. "
                    "This is a model estimate, not a guarantee."
                )

                if high_risk:
                    st.error("HIGHER RISK — Review this shipment.")
                    st.write(
                        "Recommended action: verify shipment scheduling, "
                        "carrier status, inventory readiness and possible "
                        "transport bottlenecks."
                    )
                else:
                    st.success("LOWER RISK — Continue routine monitoring.")
                    st.write(
                        "Recommended action: maintain normal tracking and "
                        "reassess if shipment conditions change."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.markdown("### Batch Order Predictions")
        st.write(
            "Upload a CSV containing the model's input features. "
            "The column names must match the required names exactly."
        )

        st.caption("Required columns: " + ", ".join(feature_columns))

        # Create a sample input file from existing records.
        sample = df[feature_columns].head(10)
        st.download_button(
            "Download Sample Input CSV",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="apl_sample_prediction_input.csv",
            mime="text/csv"
        )

        uploaded = st.file_uploader(
            "Upload order CSV",
            type=["csv"],
            key="batch_csv"
        )

        if uploaded is not None:
            try:
                try:
                    batch = pd.read_csv(uploaded, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    batch = pd.read_csv(uploaded, encoding="latin1")

                if batch.empty:
                    st.warning("The uploaded CSV contains no rows.")
                else:
                    missing = [
                        c for c in feature_columns if c not in batch.columns
                    ]

                    if missing:
                        st.error(
                            "Missing required columns: " + ", ".join(missing)
                        )
                    else:
                        model_input = prepare_model_input(
                            batch, feature_columns, df
                        )
                        probabilities = model.predict_proba(
                            model_input
                        )[:, 1]

                        results = batch.copy()
                        results["Late_Delivery_Probability"] = probabilities
                        results["Predicted_Late_Delivery_Risk"] = [
                            "Higher Risk" if p >= threshold else "Lower Risk"
                            for p in probabilities
                        ]

                        high = int((probabilities >= threshold).sum())
                        low = len(results) - high

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Orders Analyzed", f"{len(results):,}")
                        c2.metric("Higher-Risk Orders", f"{high:,}")
                        c3.metric("Lower-Risk Orders", f"{low:,}")

                        st.dataframe(results.head(100), use_container_width=True)

                        st.download_button(
                            "Download All Prediction Results",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="batch_late_delivery_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


# --------------------------------------------------
# PAGE 3: REGION AND SHIPPING ANALYSIS
# --------------------------------------------------
elif page == "Region & Shipping Analysis":
    st.header("Region & Shipping Analysis")
    st.write(
        "Explore historical late-delivery patterns across regions "
        "and shipping methods."
    )

    region_col = find_column(
        df, ["Order Region", "Region", "Order Country", "Market"]
    )
    shipping_col = find_column(
        df, ["Shipping Mode", "ShippingMode", "Shipping Type"]
    )

    if region_col:
        st.markdown(f"### Late-Delivery Rate by {region_col}")
        region_table = late_rate_table(df, region_col)
        st.dataframe(
            region_table.round(2), use_container_width=True
        )

        chart = region_table.set_index(region_col)[["Late_Rate"]]
        st.bar_chart(chart)

        st.markdown("### Order Volume by Region")
        volume = (
            df.groupby(region_col, dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        st.bar_chart(volume)

    else:
        st.warning(
            "No recognised region/country column was found. "
            "Available columns are listed below."
        )
        st.write(df.columns.tolist())

    if shipping_col:
        st.markdown(f"### Late-Delivery Rate by {shipping_col}")
        shipping_table = late_rate_table(df, shipping_col)
        st.dataframe(
            shipping_table.round(2), use_container_width=True
        )
        st.bar_chart(
            shipping_table.set_index(shipping_col)[["Late_Rate"]]
        )
    else:
        st.warning(
            "No recognised shipping-mode column was found. "
            "Available columns are listed below."
        )
        st.write(df.columns.tolist())

    st.caption(
        "Rates shown here are historical dataset statistics. "
        "They describe associations and do not prove that a region "
        "or shipping method caused a delay."
    )


# --------------------------------------------------
# PAGE 4: OPERATIONS ACTION PANEL
# --------------------------------------------------
elif page == "Operations Action Panel":
    st.header("Operations Action Panel")
    st.write(
        "Use the historical data and model predictions to support "
        "shipment monitoring and operational reviews."
    )

    st.markdown("### Current Historical Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders Reviewed", f"{total_orders:,}")
    c2.metric("Historical Late Orders", f"{late_count:,}")
    c3.metric("Historical Late Rate", f"{late_rate:.2f}%")

    st.markdown("### Recommended Operational Actions")

    st.markdown(
        """
        **1. Prioritise higher-risk shipments**
        
        Review flagged orders and verify estimated delivery dates,
        carrier status and shipment tracking updates.

        **2. Investigate recurring bottlenecks**
        
        Compare historical late-delivery rates by region and shipping
        mode. Investigate persistent patterns before changing operations.

        **3. Improve exception management**
        
        Create a daily review list for delayed or high-risk shipments
        and assign follow-up responsibility to the operations team.

        **4. Monitor model performance**
        
        Compare predictions against actual delivery outcomes regularly.
        Reassess the model and its decision threshold before operational
        deployment.
        """
    )

    st.markdown("### Highest Historical Late-Delivery Rates")

    region_col = find_column(
        df, ["Order Region", "Region", "Order Country", "Market"]
    )
    shipping_col = find_column(
        df, ["Shipping Mode", "ShippingMode", "Shipping Type"]
    )

    if region_col:
        st.markdown("**Regions / markets requiring review**")
        st.dataframe(
            late_rate_table(df, region_col).head(10).round(2),
            use_container_width=True
        )

    if shipping_col:
        st.markdown("**Shipping modes requiring review**")
        st.dataframe(
            late_rate_table(df, shipping_col).round(2),
            use_container_width=True
        )

    st.warning(
        "This dashboard is a predictive analytics prototype. "
        "Model predictions should be validated against actual outcomes "
        "before being used for live shipment decisions."
    )

    st.download_button(
        "Download Historical Late-Delivery Records",
        data=df[df[TARGET] == 1].to_csv(index=False).encode("utf-8"),
        file_name="apl_historical_late_delivery_records.csv",
        mime="text/csv"
    )


st.divider()
st.caption(
    "APL Logistics | Predictive Analytics Prototype | "
    "Historical analysis and model estimates are decision-support tools."
)
