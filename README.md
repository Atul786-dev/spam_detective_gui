# spam_detective_gui
from PIL import Image, ImageTk 
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import re
from nltk.stem import PorterStemmer
from imblearn.over_sampling import RandomOverSampler

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Spam Detector")
        self.root.geometry("800x600")
        self.root.configure(bg="#078a8f")

        self.dataset = None
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()
        self.svm_model = SVC(kernel='linear')
        self.porter_stemmer = PorterStemmer()

        # Window Icon
        icon = tk.PhotoImage(file=r"D:\Saved Pictures\spam image\icon3.png")
        root.iconphoto(False, icon)

        # **TOP IMAGE (DEFAULT)**
        tk.Label(root, text="Spam Detector", font=("Arial", 18, "bold"), fg="Black", bg="#00cc00").pack(pady=5)

        image = Image.open(r"D:\Saved Pictures\spam image\icon7.png").resize((100, 100), Image.LANCZOS)
        self.default_image = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(root, image=self.default_image, bg="#078a8f")
        self.image_label.pack(pady=5)

        self.test_entry = tk.Text(root, width=70, height=2, font=("Arial", 12),fg="Black", bg="#adad85")
        self.test_entry.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="#293d3d", bg="#078a8f")
        self.result_label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_dataset, font=("Arial", 12), bg="#61afef", fg="black")
        self.load_button.pack(pady=5)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model, state=tk.DISABLED, font=("Arial", 12), bg="#ff7733", fg="black")
        self.train_button.pack(pady=5)

        self.predict_button = tk.Button(root, text="Check Spam", command=self.predict_message, state=tk.DISABLED, font=("Arial", 12), bg="#e06c75", fg="black")
        self.predict_button.pack(pady=5)

        # RESULT LABEL (Text Output)

    
    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset = pd.read_csv(file_path, encoding='latin-1')
            self.dataset.drop_duplicates(keep='first', inplace=True)
            self.train_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Dataset Loaded Successfully!")

    def preprocessor(self, text):
        text = text.lower()
        text = re.sub("\W", " ", text)
        words = text.split()
        stemmed_words = [self.porter_stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def train_model(self):
        if self.dataset is not None:
            x = self.dataset['v2'].values
            y = self.dataset['v1'].map({'ham': 0, 'spam': 1}).values
            
            self.vectorizer = CountVectorizer(preprocessor=self.preprocessor)
            x = self.vectorizer.fit_transform(x)
            
            ros = RandomOverSampler(random_state=42)
            x, y = ros.fit_resample(x, y)
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
            
            self.clf.fit(x_train, y_train)
            self.svm_model.fit(x_train, y_train)
            
            nb_acc = accuracy_score(y_test, self.clf.predict(x_test))
            svm_acc = accuracy_score(y_test, self.svm_model.predict(x_test))
            
            messagebox.showinfo("Training Complete", f"NB Accuracy: {nb_acc:.2f}\nSVM Accuracy: {svm_acc:.2f}")
            self.predict_button.config(state=tk.NORMAL)
            
            self.plot_confusion_matrix(y_test, self.clf.predict(x_test))
        else:
            messagebox.showwarning("Error", "Please load a dataset first.")

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def predict_message(self):
        message = self.test_entry.get("1.0", "end-1c")  # Fetch text from Text widget
        if message.strip():  # Ensure message is not empty
            data = self.vectorizer.transform([message])
            prediction = self.clf.predict(data)[0]

            # Show Prediction Text
            result_text = "! Spam" if prediction == 1 else "Not Spam"
            self.result_label.config(text=f"Prediction: {result_text}", fg="#293d3d")

            # Change Image Based on Prediction
            if prediction == 1:
                image_path = r"D:\Saved Pictures\spam image\9377979.png"
            else:
                image_path = r"D:\Saved Pictures\spam image\icon2.png"

        # **Resize Image to 100x100**
            image = Image.open(image_path)  # Load Image
            image = image.resize((100, 100), Image.LANCZOS)  # Resize Image
            self.result_image = ImageTk.PhotoImage(image)  # Convert to Tkinter-compatible format

        # Show Image in Label
            self.image_label.config(image=self.result_image)
            self.image_label.image = self.result_image  # Prevent garbage collection

        else:
            messagebox.showwarning("Error", "Please enter a message to check.")

# Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()
