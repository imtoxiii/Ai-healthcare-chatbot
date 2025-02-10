import streamlit as st
from transformers import pipeline

# Load the medical text generation model
@st.cache_resource
def load_medical_model():
    return pipeline(
        "text-generation",
        model="microsoft/BioGPT",  # Medical-specific model
        max_new_tokens=200,        # Limit response length
        do_sample=True,            # Enable creative responses
        top_k=50,                  # Consider top 50 possible next words
        top_p=0.95                 # Nucleus sampling threshold
    )

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

def get_medical_response(user_input, chatbot):
    user_input = user_input.lower()
    
    for keyword in medical_responses:
        if keyword in user_input:
            return f"Medical Assistant: {medical_responses[keyword]}\n\n*Always consult a doctor*"
    
    try:
        prompt = f"Explain the casues and treatment for: {user_input}\nMedical Answer:"
        
        response = chatbot(
            prompt,
            max_length=300,
            num_return_sequences=1
        )[0]['generated_text']
        
        answer = response.split("Medical Answer:")[-1].strip()
        
        return f"Medical Assistant: {answer}\n\n*Consult a healthcare professional for personal advice*"
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("AI Medical Assistant 🩺")
    st.write("**Note:** This provides general health information, not medical advice")
    

    user_input = st.text_input("Ask your health question:", "")

    if user_input.strip():

        chatbot = load_medical_model()
        

        with st.spinner("Analyzing your question..."):
            response = get_medical_response(user_input, chatbot)
        
        st.success(response)
    else:
        st.warning("Please enter a health-related question")

if __name__ == "__main__":
    main()