import sqlite3
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import re
from textblob import TextBlob

# Load model and tokenizer
model_path = "./multi_label_intent_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Intent labels
intent_labels = ['greeting', 'track_order', 'find_by_email', 'check_return', 'return_order', 'exit']

# Connect to DB
conn = sqlite3.connect("data_order.db")
cursor = conn.cursor()

# Memory
memory = {
    'email': None,
    'order_id': None,
    'last_intents': []
}

def correct_typo(text):
    return str(TextBlob(text).correct())

def predict_intents(text):
    text = correct_typo(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    results = {label: float(prob) for label, prob in zip(intent_labels, probs)}
    predicted = [label for label, prob in results.items() if prob > 0.5]
    return predicted

def extract_email(text):
    match = re.search(r"[\w.-]+@[\w.-]+", text)
    return match.group(0) if match else None

def extract_order_id(text):
    match = re.search(r"\b\d{1,5}\b", text)
    return match.group(0) if match else None

def fetch_order_details(order_id=None, email=None):
    if order_id:
        cursor.execute("SELECT * FROM order_details WHERE order_id = ?", (order_id,))
    elif email:
        cursor.execute("SELECT * FROM order_details WHERE user_email = ?", (email,))
    else:
        return None
    return cursor.fetchall()

def respond_to_intents(predicted, text):
    global memory
    text_lower = text.lower()
    if any(word in text_lower for word in ['exit', 'bye', 'quit', 'cancel', 'no thanks', 'no, thank you']):
        memory['order_id'] = None
        memory['email'] = None
        memory['last_intents'] = []
        return "Thank you! Have a great day.", True

    is_first_greeting = 'greeting' in predicted and not memory['last_intents'] and not memory['order_id'] and not memory['email']
    predicted = [p for p in predicted if p != 'greeting']

    email = extract_email(text)
    order_id = extract_order_id(text)
    if email:
        memory['email'] = email
    if order_id:
        memory['order_id'] = order_id

    if not predicted and (memory['order_id'] or memory['email']):
        predicted = memory['last_intents'] if memory['last_intents'] else ['track_order']

    response = ""
    if is_first_greeting:
        response += "Hello! I'm PORTOS – your order assistant. How can I help you today?\n"

    if 'track_order' in predicted or 'find_by_email' in predicted:
        if any(word in text_lower for word in ['another', 'new']):
            memory['order_id'] = None
            memory['email'] = None
            memory['last_intents'] = predicted  # Update intent to maintain context
            return "Sure, could you provide the new order ID or email?", False
        if not memory['order_id'] and not memory['email']:
            return "Could you provide your order ID or email so I can help you?", False
        else:
            orders = fetch_order_details(memory['order_id'], memory['email'])
            if orders:
                for order in orders:
                    response += (
                        f"User Name: {order[1]}\n"
                        f"Email ID: {order[2]}\n"
                        f"Product Name: {order[3]}\n"
                        f"Quantity: {order[4]}\n\n"
                    )
            else:
                response += "No order found with the given details.\n"

    if 'check_return' in predicted:
        if memory['order_id'] or memory['email']:
            orders = fetch_order_details(memory['order_id'], memory['email'])
            if orders:
                for order in orders:
                    response += f"Return Status: {order[5] or 'No return initiated'}"
            else:
                response += "No return information found for the given details."
        else:
            return "Could you provide your order ID or email so I can help you check the return status?", False
            response += "Please provide your order ID or email to check return status.\n"

    if 'return_order' in predicted:
        if memory['order_id'] or memory['email']:
            orders = fetch_order_details(memory['order_id'], memory['email'])
            if orders:
                for order in orders:
                    status = order[5]
                    if status is None:
                        cursor.execute("UPDATE order_details SET order_return_status = 'Processing' WHERE order_id = ?", (order[0],))
                        conn.commit()
                        response += "Your return has been initiated."
                    elif status == 'Rejected':
                        response += "Sorry, your return has been rejected."
                    else:
                        response += f"Your return is already {status.lower()}."
            else:
                response += "No returnable order found with the given details."
        else:
            return "Please provide your order ID or email so I can process the return.", False
        response += f"Your return is already {status.lower()}.\n"

    if response.strip() and any(intent in predicted for intent in ['track_order', 'find_by_email', 'check_return', 'return_order']) and not response.strip().endswith('Is there anything else I can help you with?'):
        response += "Is there anything else I can help you with?\n"

    memory['last_intents'] = predicted
    return response.strip(), False

# Chat loop
print("PORTOS 🤖: Hello! I'm PORTOS – your order assistant. How can I help you today? (Type 'exit' to leave)")
while True:
    user_input = input("You: ")
    predicted_intents = predict_intents(user_input)
    response, should_exit = respond_to_intents(predicted_intents, user_input)
    print(f"PORTOS 🤖: {response}")
    if should_exit:
        break
