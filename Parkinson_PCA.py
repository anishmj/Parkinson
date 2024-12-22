import numpy as np
import pickle
import streamlit as st

# Load the scaler and model
scaler = pickle.load(open('scaler.sav', 'rb'))
loaded_model = pickle.load(open('parkinsons_hybrid_model.sav', 'rb'))

def main():
    st.title("Parkinson Prediction Web App")
    
    # Create input fields for the required 8 features using two columns
    col1, col2 = st.columns(2)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', '0')
        fhi = st.text_input('MDVP:Fhi(Hz)', '0')
        flo = st.text_input('MDVP:Flo(Hz)', '0')
        Jitter_percent = st.text_input('MDVP:Jitter(%)', '0')

    with col2:
        Shimmer = st.text_input('MDVP:Shimmer', '0')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', '0')
        APQ = st.text_input('MDVP:APQ', '0')
        NHR = st.text_input('NHR', '0')

    # Code for Prediction
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):
        try:
            # Convert inputs to float and create a feature array
            features = [float(fo), float(fhi), float(flo), 
                        float(Jitter_percent), float(Shimmer), 
                        float(Shimmer_dB), float(APQ), 
                        float(NHR)]
            
            # Check if any input value is zero
            if any(value == 0 for value in features):
                st.error("Please enter correct vocal value")
            else:
                # Scale the features
                scaled_features = scaler.transform([features])
                
                # Make a prediction using the loaded model
                prediction = loaded_model.predict(scaled_features)
                
                # Convert prediction to a readable format
                parkinsons_diagnosis = 'Positive for Parkinson\'s Disease' if prediction[0] == 1 else 'Negative for Parkinson\'s Disease'
                
                # Display the prediction result
                st.success(parkinsons_diagnosis)
        
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")
    
if __name__ == '__main__':
    main()
