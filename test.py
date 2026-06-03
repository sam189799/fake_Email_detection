import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

email = input("Enter Email Text: ")

email_vector = vectorizer.transform([email])

result = model.predict(email_vector)

if result[0] == 1:
    print("Fake Email / Spam")
else:
    print("Genuine Email")