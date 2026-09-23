import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = {
    'email_text': [
        "Urgent: Your bank account has been compromised. Click here to verify your identity: http://secure-update-bank.com",
        "Hey Bob, are we still on for lunch tomorrow at 12?",
        "Win a free iPhone 14! Click the link below to claim your prize now!! http://free-gifts-now.info",
        "Please review the attached quarterly financial report before the meeting.",
        "Warning: Your Office365 password expires in 2 hours. Update immediately at http://login-office365-update.com",
        "Can you send me the presentation slides from yesterday's meeting? Thanks.",
        "Congratulations! You've been selected for a $1000 Walmart gift card. Claim here: http://walmart-winner.com",
        "Mom, I'll be home late today. Don't wait up for dinner.",
        "Action Required: Unpaid invoice #9824. Please pay immediately to avoid account suspension. Link: http://invoice-payment.com",
        "Let's schedule a quick sync to discuss the upcoming project deliverables."
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}


df = pd.DataFrame(data)


X = df['email_text']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


y_pred = model.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Safe (0)", "Phishing (1)"])

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print("[[True Negatives  False Positives]")
print(" [False Negatives True Positives]]")
print(conf_matrix, "\n")

print("Classification Report:")
print(class_report)


new_email = ["Verify your account details here to avoid lockout: http://fake-login-alert.com"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)

print("--- Testing New Email ---")
print(f"Email: {new_email[0]}")
print("Prediction:", "Phishing 🚨" if prediction[0] == 1 else "Safe ✅")