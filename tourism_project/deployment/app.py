import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model

# replace with your repoid
model_path = hf_hub_download(repo_id="rakeshambudkar/machine_failure_model", filename="best_machine_failure_model_v1.joblib")

model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a tourism package purchase based on its operational parameters.
Please enter the details in below to get a prediction.
""")

# User 
Age= st.number_input("Age", min_value=18, value=32)
CityTier= st.number_input("CityTier", min_value=1, value=2)
DurationOfPitch= st.number_input("DurationOfPitch", min_value=1, value=50)
NumberOfFollowups= st.number_input("NumberOfFollowups", min_value=1, value=6)
PreferredPropertyStar= st.number_input("PreferredPropertyStar", min_value=1, value=2)
NumberOfTrips= st.number_input("NumberOfTrips", min_value=1, value=5)
Passport= st.number_input("Passport", min_value=0, value=0)
PitchSatisfactionScore= st.number_input("PitchSatisfactionScore", min_value=1, value=2)
OwnCar= st.number_input("OwnCar", min_value=1, value=1)
NumberOfChildrenVisiting= st.number_input("NumberOfChildrenVisiting", min_value=1, value=2)
MonthlyIncome= st.number_input("MonthlyIncome", min_value=1, value=2000000)

TypeofContact= st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])
Occupation=st.selectbox("TypeofContact", ["Salaried", "Free Lancer","Small Business","Large Business"])
Gender=st.selectbox("TypeofContact", ["Male", "Female"])
ProductPitched=st.selectbox("TypeofContact", ["Basic", "Standard","Deluxe","Super Deluxe","King"])
MaritalStatus=st.selectbox("TypeofContact", ["Married", "Single","Divorced"])
Designation=st.selectbox("TypeofContact", ["Executive","Manager","Senior Manager","AVP","VP"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport':Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation

}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
