import streamlit as st
from transformers import pipeline

# Load medical QA model
@st.cache_resource
def load_medical_model():
    return pipeline("text-generation", model="microsoft/BioGPT")

# Enhanced healthcare response logic
def get_medical_response(user_input, chatbot):
    user_input = user_input.lower()
    
    # Medical knowledge base
    medical_responses = {
        "fever": "For fever: Rest, hydrate, and monitor temperature. Use acetaminophen/ibuprofen. Seek help if >103°F or lasting >3 days.",
        "headache": "For headaches: Ensure hydration, rest in dark room. Use OTC pain relievers. Seek help for sudden severe pain.",
        "cough": "For coughs: Stay hydrated. Honey/lemon may help. Consult doctor if persistent >2 weeks or with blood.",
        "covid": "COVID-19 symptoms include fever, cough, fatigue. Isolate and test. High-risk patients should seek antivirals.",
        "diabetes": "Monitor blood sugar regularly. Follow prescribed insulin/diet. Watch for hypoglycemia symptoms.",
        "pregnant": "Schedule prenatal care immediately. Avoid alcohol/smoking. Report any bleeding/severe cramps.",
        "allergy": "For allergies: Use antihistamines for mild reactions. Carry epinephrine for severe reactions.",
        "burn": "Cool burns under water for 10 mins. Don't use ice. Cover with sterile dressing. Seek help for large burns.",
        "asthma": "Use prescribed inhalers. Avoid triggers. Seek ER for unrelieved breathing difficulty.",
        "vaccine": "Vaccines available for flu, COVID, HPV, etc. Consult local health center for schedules."
    }

    # Check direct matches
    for term in medical_responses:
        if term in user_input:
            return medical_responses[term]

    # Generate response for other queries
    response = chatbot(
        user_input,
        max_length=150,
        num_return_sequences=1,
        do_sample=True
    )[0]['generated_text']
    
    # Ensure clinical relevance
    return f"Medical Assistant: {response}\n\n*Always consult a doctor for personal medical advice*"

# Streamlit interface
def main():
    st.title("Medical QA Assistant 🩺")
    st.write("**Note:** General health information only - not professional medical advice")
    
    user_input = st.text_input("Ask any health-related question:", "")
    
    if user_input.strip():
        chatbot = load_medical_model()
        with st.spinner("Consulting medical knowledge..."):
            response = get_medical_response(user_input, chatbot)
        st.success(response)
    else:
        st.warning("Please enter your health question")

if __name__ == "__main__":
    main()