
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="MTCT Risk Predictor", page_icon="🍼", layout="centered")

# --- Helper functions to extract things from pipeline ---
def load_pipeline(path="rf_pipeline.pkl"):
    try:
        with open(path, "rb") as f:
            pipe = pickle.load(f)
        return pipe
    except FileNotFoundError:
        st.error(f"Pipeline file not found: {path}. Place rf_pipeline.pkl next to this app.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()

def get_model_from_pipeline(pipe):
    if hasattr(pipe, "named_steps"):
        if "model" in pipe.named_steps:
            return pipe.named_steps["model"]
        for step in pipe.named_steps.values():
            if isinstance(step, RandomForestClassifier):
                return step
    if isinstance(pipe, RandomForestClassifier):
        return pipe
    return None

def extract_feature_names_from_preprocessor(pipe):
    pre = None
    if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
        pre = pipe.named_steps["preprocessor"]
    try:
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            names = pre.get_feature_names_out()
            return list(names)
    except Exception:
        pass

    try:
        if pre is not None and hasattr(pre, "transformers_"):
            names = []
            for name, transformer, cols in pre.transformers_:
                if transformer == "drop":
                    continue
                if transformer == "passthrough":
                    if isinstance(cols, (list, tuple)):
                        names.extend(list(cols))
                    else:
                        names.append(cols)
                else:
                    try:
                        if hasattr(transformer, "get_feature_names_out"):
                            out = transformer.get_feature_names_out(cols)
                            names.extend(list(out))
                        elif hasattr(transformer, "categories_"):
                            for i, col in enumerate(cols):
                                cats = transformer.categories_[i]
                                for cat in cats:
                                    names.append(f"{col}_{cat}")
                        else:
                            if isinstance(cols, (list, tuple)):
                                names.extend(list(cols))
                            else:
                                names.append(cols)
                    except Exception:
                        if isinstance(cols, (list, tuple)):
                            names.extend(list(cols))
                        else:
                            names.append(cols)
            if len(names) > 0:
                return list(names)
    except Exception:
        pass
    return None

# --- Load pipeline and model ---
rf_pipe = load_pipeline("rf_pipeline.pkl")
rf_model = get_model_from_pipeline(rf_pipe)

feature_names = None
try:
    feature_names = pickle.load(open("feature_cols.pkl", "rb"))
except Exception:
    feature_names = extract_feature_names_from_preprocessor(rf_pipe)

if feature_names is None:
    st.warning("Could not automatically determine preprocessed feature names (feature importance will be unavailable).")

# --- CSS decoration ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg,#f7fbff,#fffaf6);
    }
    .big-font { font-size:20px !important; }
    .small-muted { color: #6b7280; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center'>🍼 MTCT Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='small-muted' style='text-align:center'>Predict probability of HIV transmission to infant. Use as decision support — not a diagnostic tool.</p>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Probability threshold for 'High Risk'", min_value=0.01, max_value=0.9, value=0.30, step=0.01)
clinical_override = st.sidebar.checkbox("Apply clinical override (viral load >1000 → High Risk)", value=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["🔮 Predict", "📈 Model Insights"])

with tab1:
    st.header("Predict MTCT Risk")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            last_vl = st.number_input("Maternal Viral Load (copies/mL)", min_value=0, value=1000, step=1)
            art_duration = st.number_input("ART Duration (months)", min_value=0, value=12, step=1)
            months_prescription = st.number_input("Months of Prescription (refill interval)", min_value=0, value=3, step=1)
        with col2:
            active_pmtct = st.selectbox("Active in PMTCT program?", ["No", "Pregnant", "Breastfeeding"], index=1)
            tpt_outcome = st.selectbox("Prophylaxis/TPT Outcome", ["Not Started", "Ongoing", "Treatment completed", "Discontinued"], index=2)
            note = st.checkbox("Show interpretation bullets", value=True)
        submitted = st.form_submit_button("Predict MTCT Risk")

    if submitted:
        input_df = pd.DataFrame([{
            "last vl": float(last_vl),
            "art_duration_months": float(art_duration),
            "months of prescription": float(months_prescription),
            "active in pmtct": active_pmtct,
            "tpt outcome": tpt_outcome
        }])

        try:
            proba = rf_pipe.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        model_classes = getattr(rf_pipe, "classes_", None)
        idx_high = -1
        if model_classes is not None:
            try:
                idx_high = list(model_classes).index("High Risk")
            except Exception:
                if len(model_classes) == 2:
                    idx_high = 1
        prob_high = float(proba[idx_high])

        ml_label = "High Risk" if prob_high >= threshold else "Low Risk"
        final_label = "High Risk" if (clinical_override and last_vl > 1000) else ml_label

        st.markdown("### Result")
        if final_label == "High Risk":
            st.error(f"**High Risk** — Probability: **{prob_high*100:.2f}%**")
        else:
            st.success(f"**Low Risk** — Probability: **{prob_high*100:.2f}%**")

        if note:
            st.markdown("### 🔍 Interpretation")
            if last_vl > 1000:
                st.write("- ⚠️ High maternal viral load (>1000) increases MTCT risk.")
            else:
                st.write("- ✅ Viral load suppressed/low.")
            if art_duration < 6:
                st.write("- ⚠️ ART <6 months increases risk.")
            else:
                st.write("- ✅ Adequate ART duration.")
            if active_pmtct == "No":
                st.write("- ⚠️ Not in PMTCT program.")
            else:
                st.write(f"- ✅ Active in PMTCT ({active_pmtct}).")
            if tpt_outcome in ["Not Started", "Discontinued"]:
                st.write("- ⚠️ Prophylaxis/TPT not completed.")
            else:
                st.write(f"- ✅ Prophylaxis status: {tpt_outcome}.")

with tab2:
    st.header("Model Insights")
    if rf_model is None:
        st.warning("Could not extract model from pipeline.")
    else:
        if feature_names is None:
            st.info("Feature names unavailable. Showing raw feature importances.")
            try:
                importances = rf_model.feature_importances_
                fig, ax = plt.subplots()
                ax.barh(range(len(importances)), importances)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            try:
                importances = rf_model.feature_importances_
                fi = pd.DataFrame({"feature": feature_names, "importance": importances})
                fi = fi.sort_values("importance", ascending=True).tail(15)
                fig, ax = plt.subplots(figsize=(8,5))
                ax.barh(fi["feature"], fi["importance"])
                st.pyplot(fig)
                st.write(fi.sort_values("importance", ascending=False).head(10).reset_index(drop=True))
            except Exception as e:
                st.error(f"Error computing importances: {e}")

    st.markdown("---")
    st.subheader("Pipeline steps")
    if hasattr(rf_pipe, "named_steps"):
        for k, v in rf_pipe.named_steps.items():
            st.write(f"- **{k}**: {type(v).__name__}")
    else:
        st.write("Pipeline does not expose named_steps.")

st.markdown("---")
st.markdown("<div style='text-align:center;font-size:12px;color:#6b7280'>Built for PMTCT decision support — verify with clinical staff before acting on predictions.</div>", unsafe_allow_html=True)
