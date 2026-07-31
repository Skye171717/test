import os
import textwrap
import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered"
)

## --------------------------------------------------------------------------
## Visual styling (colors, font, card layout) - no emojis, images used instead
## The color palette here is intentionally also fixed in .streamlit/config.toml
## so the app always renders the same, regardless of the visitor's system
## dark-mode setting.
## --------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #eef5fb 0%, #ffffff 45%);
}

.app-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #1b3a57;
    margin-bottom: 0.1rem;
}

.app-subtitle {
    color: #55677a;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.section-card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 10px rgba(27, 58, 87, 0.07);
    border: 1px solid #e7edf3;
    margin-bottom: 1.4rem;
}

.section-heading {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1b3a57;
    margin-bottom: 0.8rem;
    border-left: 4px solid #4a90c4;
    padding-left: 0.6rem;
}

.summary-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}

.summary-chip {
    background-color: #f2f7fb;
    border: 1px solid #dce7f0;
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    color: #1b3a57;
}

.result-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.6rem;
    border: 1px solid;
}

.result-card.low-risk {
    background-color: #eaf7ef;
    border-color: #bfe3cc;
}

.result-card.high-risk {
    background-color: #fbebea;
    border-color: #f2c4c1;
}

.result-heading {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.result-heading.low-risk-text {
    color: #216e46;
}

.result-heading.high-risk-text {
    color: #a5322c;
}

.result-caption {
    color: #55677a;
    font-size: 0.85rem;
    margin-top: 0.6rem;
}

.progress-track {
    background-color: #e4e9ee;
    border-radius: 8px;
    height: 12px;
    width: 100%;
    margin-top: 0.5rem;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 8px;
}

.progress-fill.low-risk-fill {
    background-color: #3fa06a;
}

.progress-fill.high-risk-fill {
    background-color: #d1554d;
}

.next-steps-card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-top: 0.9rem;
    border: 1px solid #e7edf3;
}

.next-steps-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #1b3a57;
    margin-bottom: 0.5rem;
}

.next-steps-card ul {
    margin: 0;
    padding-left: 1.2rem;
    color: #3a4a5a;
    font-size: 0.9rem;
}

.next-steps-card li {
    margin-bottom: 0.35rem;
}

.sidebar-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    border: 1px solid #e7edf3;
    margin-bottom: 0.9rem;
}

.sidebar-card-heading {
    font-weight: 600;
    color: #1b3a57;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid #4a90c4;
    padding-left: 0.5rem;
}

.sidebar-card p, .sidebar-card li {
    color: #55677a;
    font-size: 0.85rem;
}

div.stButton > button {
    background-color: #1b3a57;
    color: #ffffff;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
}

div.stButton > button:hover {
    background-color: #2c5578;
    color: #ffffff;
}
</style>
"""), unsafe_allow_html=True)

## Load trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("decision_tree_model.pkl")
    except FileNotFoundError:
        return None

model = load_model()

## --------------------------------------------------------------------------
## Page header
## An optional banner image can be dropped alongside this script.
## If it isn't present, the app simply skips it - no broken image, no error.
## --------------------------------------------------------------------------
banner_path = "banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

st.markdown('<div class="app-title">Diabetes Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Estimate the likelihood of diabetes based on your health profile.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("Could not find `decision_tree_model.pkl`. Please place the model file alongside this script.")
    st.stop()

## --------------------------------------------------------------------------
## Sidebar - restyled into short, scannable cards instead of one text block
## --------------------------------------------------------------------------
with st.sidebar:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)

    st.markdown(textwrap.dedent("""
        <div class="sidebar-card">
            <div class="sidebar-card-heading">How it works</div>
            <p>Fill in the patient details, then select
            <strong>Predict Diabetes Risk</strong> to get an instant
            screening result and estimated probability.</p>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-card-heading">About the model</div>
            <p>This app uses a Decision Tree model trained on health
            screening data, prioritised to catch as many at-risk cases
            as possible.</p>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-card-heading">Important note</div>
            <p>This is a screening tool, not a medical diagnosis. Always
            confirm results with a qualified healthcare professional.</p>
        </div>
    """), unsafe_allow_html=True)

## Define input options
genders = ["Female", "Male"]
smoking_histories = ["Past Smoker", "Current Smoker", "Non-Smoker", "Prefer Not to Say"]

## --------------------------------------------------------------------------
## Patient information - grouped in a card
## --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender_selected = st.selectbox("Gender", genders)
    age_selected = st.number_input(
        "Age (years)", min_value=0, max_value=120, value=30, step=1,
        help=(
            "The prediction model was trained on individuals aged up to "
            "80. Results for ages above 80 are shown, but treat them as "
            "an extrapolation with more uncertainty."
        )
    )
    if age_selected > 80:
        st.caption(
            "Note: this model was trained on data up to age 80. "
            "Results above that age are less reliable."
        )
    hypertension_selected = st.radio(
        "Does the patient have hypertension?", ["No", "Yes"], horizontal=True
    )
    heart_disease_selected = st.radio(
        "Does the patient have heart disease?", ["No", "Yes"], horizontal=True
    )

with col2:
    smoking_history_selected = st.selectbox("Smoking History", smoking_histories)

    height_selected = st.number_input(
        "Height (cm)", min_value=100.0, max_value=250.0, value=165.0, step=0.5
    )
    weight_selected = st.number_input(
        "Weight (kg)", min_value=20.0, max_value=250.0, value=65.0, step=0.5
    )

    ## BMI is calculated automatically from height and weight rather than
    ## asking the user to know or enter it directly
    bmi_selected = weight_selected / ((height_selected / 100) ** 2)
    st.caption(f"Calculated BMI: {bmi_selected:.1f}")

    hba1c_selected = st.slider(
        "HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1,
        help=(
            "HbA1c reflects your average blood sugar level over the past "
            "2 to 3 months, shown as a percentage. Roughly: below 5.7% is "
            "normal, 5.7 to 6.4% is prediabetes, and 6.5% or above is in "
            "the diabetes range."
        )
    )
    glucose_selected = st.slider(
        "Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=100, step=1,
        help=(
            "Blood glucose is the amount of sugar in your blood at the "
            "moment it's measured. Unlike HbA1c, which shows a longer-term "
            "average, this reflects a single point in time."
        )
    )

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("What do HbA1c and blood glucose mean, and where can I get tested?"):
    st.markdown(
        "- **HbA1c (%)** - a blood test result showing your average blood "
        "sugar over roughly the last 2 to 3 months. It's used because a "
        "single blood sugar reading can vary a lot depending on when you "
        "last ate, while HbA1c reflects a longer-term trend.\n"
        "- **Blood glucose (mg/dL)** - the amount of sugar circulating in "
        "your blood at the time of testing. It can rise and fall quickly, "
        "for example after a meal, so it's a snapshot rather than an average.\n"
        "\n"
        "**Getting these readings in Singapore**\n"
        "- **HbA1c** is measured through a blood test ordered by a doctor "
        "- available at GP clinics, polyclinics, or through the "
        "**Screen for Life** national screening programme (subsidised "
        "for eligible Singaporeans and PRs). It is not typically "
        "available as a home self-test.\n"
        "- **Blood glucose** can also be tested at a clinic or polyclinic, "
        "or self-tested at home using a glucometer / blood glucose test "
        "kit, available at pharmacies such as Guardian, Watsons and "
        "Unity, or online. A home reading is convenient for a quick "
        "check, but a clinic test is more reliable for medical decisions."
    )

## Predict button
predict_clicked = st.button("Predict Diabetes Risk", use_container_width=True)

if predict_clicked:

    ## Basic input validation with user-facing error messages
    validation_errors = []

    if age_selected <= 0:
        validation_errors.append("Age must be greater than 0.")
    if height_selected <= 0:
        validation_errors.append("Height must be greater than 0.")
    if weight_selected <= 0:
        validation_errors.append("Weight must be greater than 0.")
    if hba1c_selected <= 0:
        validation_errors.append("HbA1c level must be greater than 0.")
    if glucose_selected <= 0:
        validation_errors.append("Blood glucose level must be greater than 0.")

    if validation_errors:
        for err in validation_errors:
            st.error(err)
    else:
        try:
            with st.spinner("Running prediction..."):

                ## Map Yes/No radio inputs back to the 0/1 encoding used in training
                hypertension_value = 1 if hypertension_selected == "Yes" else 0
                heart_disease_value = 1 if heart_disease_selected == "Yes" else 0

                ## Build a single-row DataFrame from the user inputs
                df_input = pd.DataFrame({
                    "gender": [gender_selected],
                    "age": [age_selected],
                    "hypertension": [hypertension_value],
                    "heart_disease": [heart_disease_value],
                    "smoking_history": [smoking_history_selected],
                    "bmi": [bmi_selected],
                    "HbA1c_level": [hba1c_selected],
                    "blood_glucose_level": [glucose_selected],
                })

                ## One-hot encode categorical columns
                df_input = pd.get_dummies(
                    df_input, columns=["gender", "smoking_history"], drop_first=True
                )

                ## Align columns with the features the model was trained on
                df_input = df_input.reindex(
                    columns=model.feature_names_in_, fill_value=0
                )

                ## Generate prediction
                prediction = model.predict(df_input)[0]

                ## Probability, if the model supports it
                diabetes_prob = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df_input)[0]
                    diabetes_prob = proba[1] if len(proba) > 1 else proba[0]

            ## ------------------------------------------------------------------
            ## Quick recap of what was entered, so the result is easy to trace
            ## back to the inputs
            ## ------------------------------------------------------------------
            summary_html = textwrap.dedent(f"""
            <div class="summary-row">
                <div class="summary-chip">Age: {age_selected}</div>
                <div class="summary-chip">Gender: {gender_selected}</div>
                <div class="summary-chip">BMI: {bmi_selected:.1f}</div>
                <div class="summary-chip">HbA1c: {hba1c_selected:.1f}%</div>
                <div class="summary-chip">Glucose: {glucose_selected} mg/dL</div>
            </div>
            """).strip()
            st.markdown(summary_html, unsafe_allow_html=True)

            ## ------------------------------------------------------------------
            ## Display results as a single styled card (color communicates risk
            ## instead of an emoji)
            ## ------------------------------------------------------------------
            risk_class = "high-risk" if prediction == 1 else "low-risk"
            risk_text_class = "high-risk-text" if prediction == 1 else "low-risk-text"
            fill_class = "high-risk-fill" if prediction == 1 else "low-risk-fill"

            result_message = (
                "The model predicts this patient may have diabetes."
                if prediction == 1
                else "The model predicts this patient is unlikely to have diabetes."
            )

            probability_html = ""
            if diabetes_prob is not None:
                pct = diabetes_prob * 100
                probability_html = textwrap.dedent(f"""
                <div style="margin-top:0.8rem; font-size:0.9rem; color:#3a4a5a;">
                    Estimated probability of diabetes: <strong>{pct:.1f}%</strong>
                </div>
                <div class="progress-track">
                    <div class="progress-fill {fill_class}" style="width:{pct:.1f}%;"></div>
                </div>
                """).strip()

            result_html = textwrap.dedent(f"""
            <div class="result-card {risk_class}">
                <div class="result-heading {risk_text_class}">{result_message}</div>
                {probability_html}
                <div class="result-caption">
                    This prediction is generated by a machine learning model and
                    should not be used as a substitute for professional medical diagnosis.
                </div>
            </div>
            """).strip()
            st.markdown(result_html, unsafe_allow_html=True)

            ## ------------------------------------------------------------------
            ## What should I do next - guidance tailored to the risk level
            ## ------------------------------------------------------------------
            if prediction == 1:
                next_steps_html = textwrap.dedent("""
                <div class="next-steps-card">
                    <div class="next-steps-heading">What should I do next?</div>
                    <ul>
                        <li>Arrange a medical assessment with a doctor or
                        polyclinic to confirm your diabetes status.</li>
                        <li>Bring these results along to help guide the
                        conversation with your healthcare provider.</li>
                        <li>Avoid self-diagnosing or self-medicating based
                        on this screening result alone.</li>
                    </ul>
                </div>
                """).strip()
            else:
                next_steps_html = textwrap.dedent("""
                <div class="next-steps-card">
                    <div class="next-steps-heading">What should I do next?</div>
                    <ul>
                        <li>Continue maintaining a healthy, balanced diet
                        and regular physical activity.</li>
                        <li>Go for routine health screenings, such as
                        Screen for Life, as recommended for your age
                        group.</li>
                        <li>Keep an eye on changes in weight, thirst, or
                        energy levels, and check in with a doctor if
                        anything feels off.</li>
                    </ul>
                </div>
                """).strip()
            st.markdown(next_steps_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong while generating the prediction. Details: {e}")
