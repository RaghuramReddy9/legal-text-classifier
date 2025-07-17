from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix




# Sample labeled legal data
texts = [
    "I want to cancel my subscription",
    "Cancel and get money back",
    "I’m not happy, cancel my plan",
    "How do I get a refund?",
    "Can I receive a refund after cancellation?",
    "Refund request for last month’s fee",
    "What are the terms and conditions?",
    "Give me the contract terms",
    "Explain the service agreement"
]

labels = [
    "cancellation",
    "cancellation",
    "cancellation",
    "refund",
    "refund",
    "refund",
    "policy",
    "policy",
    "policy"
]


# 1. Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.4, stratify=labels, random_state=42)


# 2. Vectorize text
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# 3. Train model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test_vect)

# 5. Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=["cancellation", "refund", "policy"])


print("✅ Accuracy:", acc)
print("\n📊 Confusion Matrix:\n", cm)



import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")  # or try "QtAgg" or "TkCairo"



# Class labels in same order as confusion matrix
class_labels = ["cancellation", "refund", "policy"]

# Plot confusion matrix as heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


