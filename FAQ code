import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

translator = pipeline("translation", model="facebook/m2m100_418M")
detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
summary_pipline = pipeline("summarization", model="facebook/bart-large-cnn") # המודל שהיה חסר
sentence_transformer_emcoder = SentenceTransformer('all-mpnet-base-v2')




faq_data = [
    {
        "queries": [
            "Where is my order?",
            "Track my package",
            "Order status",
            "I haven't received my delivery yet",
            "Where is my stuff?"
        ],
        "answer": "You can check the status of your order by visiting the 'My Orders' section of your account."
    },
    {
        "queries": [
            "How can I return a product?",
            "I want a refund",
            "How to send back an item?",
            "Return policy",
            "Return center"
        ],
        "answer": "To return a product, please visit our Returns Center and follow the instructions. You will need your order number and the email address associated with your account."
    },
    {
        "queries": [
            "Is the new XYZ smartphone available in stock?",
            "XYZ smartphone availability",
            "Can I buy the XYZ phone now?",
            "XYZ stock update"
        ],
        "answer": "The XYZ smartphone is currently in stock and available for immediate shipping."
    },
    {
        "queries": [
            "How long does shipping take?",
            "Delivery time",
            "When will my package arrive?",
            "Shipping speed options",
            "How fast is shipping?"
        ],
        "answer": "Standard shipping typically takes 3-5 business days. Expedited shipping options are available at checkout."
    },
    {
        "queries": [
            "I forgot my password. How can I reset it?",
            "Reset password link",
            "Can't login",
            "Change my password",
            "Account access recovery"
        ],
        "answer": "You can reset your password by clicking on 'Forgot Password' at the login page. A reset link will be sent to your registered email address."
    },
    {
        "queries": [
            "How can I contact customer support?",
            "Talk to a human",
            "Customer service phone number",
            "Support email",
            "Help desk"
        ],
        "answer": "You can contact our customer support team via the 'Contact Us' page or by calling our hotline at 1-800-SHOPEASE."
    }
]

translator = pipeline("translation", model="facebook/m2m100_418M")
detector=pipeline("text-classification",model="papluca/xlm-roberta-base-language-detection")
def client_question_to_en(text):

  detection_result = detector(text)
  detected_lang = detection_result[0]['label']

  if detected_lang == 'en':
      return text,"en"


  elif detected_lang in ['fr', 'es']:
        translation = translator(text, src_lang=detected_lang,tgt_lang="en")
        return translation[0]['translation_text'],detected_lang


  else:
        error_msg = "Sorry, the system only supports English, French, and Spanish."


        translated_error = translator(error_msg,src_lang="en",tgt_lang=detected_lang)
        return translated_error[0]['translation_text'],detected_lang

def summarize_text(text):
  if len(text.split()) <= 12:
    return text
  else:
    summary=summary_pipline(text,min_length=10,max_length=15,do_sample=False)
    return summary[0]['summary_text']



import torch
from sentence_transformers import SentenceTransformer


sentence_transformer_emcoder = SentenceTransformer('all-mpnet-base-v2')


all_questions = []
question_to_faq_map = []

for idx, item in enumerate(faq_data):
    for q in item['queries']:
        all_questions.append(q)
        question_to_faq_map.append(idx)

question_embeddings = sentence_transformer_emcoder.encode(all_questions, convert_to_tensor=True)

print("Success!")
print(f"Total variations encoded: {question_embeddings.shape[0]}")
print(f"Vector size: {question_embeddings.shape[1]}")

from sentence_transformers import util

def find_best_answer(summary_text):
  summarized_vector=sentence_transformer_emcoder.encode(summary_text,convert_to_tensor=True)
  similarities = util.cos_sim(summarized_vector, question_embeddings)[0]
  best_question_idx=similarities.argmax().item()

  confidence_score = similarities[best_question_idx].item()
  if confidence_score < 0.6:
      return "I'm sorry, I couldn't find a precise answer to your question."

  best_faq_data_idx = question_to_faq_map[best_question_idx]
  return faq_data[best_faq_data_idx]['answer']

def return_client_text(client_text):
  en_client_text, detected_lang = client_question_to_en(client_text)
  if(detected_lang in ['en','es','fr']):


    summarized_text=summarize_text(en_client_text)

    best_answer=find_best_answer(summarized_text)

    if(detected_lang=='en'):
      return best_answer


    final_translation = translator(best_answer, src_lang="en", tgt_lang=detected_lang)

    return final_translation[0]['translation_text']

  return  en_client_text

client_question = "comment puis-je vérifier ma commande" #@param {type:"string"}
return_client_text(client_question)






