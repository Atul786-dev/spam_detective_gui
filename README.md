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
        self.root.geometry("950x600")
        self.root.configure(bg="#cc0000")

        self.dataset = None
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()
        self.svm_model = SVC(kernel='linear')
        self.porter_stemmer = PorterStemmer()
        self.history = []

        icon = tk.PhotoImage(file=r"D:\\Saved Pictures\\spam image\\icon3.png")
        root.iconphoto(False, icon)

        bg_image_path = r"D:\\Saved Pictures\\spam image\\bg.png"
        bg_image = Image.open(bg_image_path).resize((1600, 800), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        show_label = tk.Label(root, text="Spam Detector", font=("Arial", 18, "bold"), fg="white", bg="#065d85")
        show_label.place(x=95, y=60)
        

        image = Image.open(r"D:\\Saved Pictures\\spam image\\s2.png").resize((100, 100), Image.LANCZOS)
        self.default_image = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(root, image=self.default_image, bg="#065d85")
        self.image_label.pack(pady=5)
        self.image_label.place(x=590,y=30)

        # Text Entry with Placeholder
        self.placeholder_text = "Enter your message here..."
        self.test_entry = tk.Text(root, width=53, height=16, font=("Arial", 12), fg="gray", bg="white")
        self.test_entry.insert("1.0", self.placeholder_text)
        self.test_entry.place(x=400, y=150)

        # Bind placeholder behavior
        self.test_entry.bind("<FocusIn>", self.clear_placeholder)
        self.test_entry.bind("<FocusOut>", self.add_placeholder_if_empty)


        self.result_label = tk.Label(root, text="result.......", width=25, height=2 , font=("Arial", 14, "bold"), fg="white", bg="black")
        self.result_label.pack(pady=10)
        self.result_label.place (x=400,y=450)

        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_dataset, font=("Arial", 14), bg="#00ace6", fg="black")
        self.load_button.pack(pady=5)
        self.load_button.place(x=120 ,y=150)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model, state=tk.DISABLED, font=("Arial", 14), bg="#ac00e6", fg="black")
        self.train_button.pack(pady=5)
        self.train_button.place(x=121 , y=220)

        self.predict_button = tk.Button(root, text="Check Spam", command=self.predict_message, state=tk.DISABLED, font=("Arial", 18), bg="#ff1a1a", fg="black")
        self.predict_button.pack(pady=5)
        self.predict_button.place(x=720,y=450)

        self.history_button = tk.Button(root, text="View History", command=self.view_history, font=("Arial", 14), bg="#ff471a", fg="black")
        self.history_button.pack(pady=5)
        self.history_button.place(x=120,y=290)

        self.save_button = tk.Button(root, text="Save Results", command=self.save_results, font=("Arial", 14), bg="#00b33c", fg="black")
        self.save_button.pack(pady=5)
        self.save_button.place(x=120,y=360)

        self.simulate_button = tk.Button(
        root,
        text="Upload Message from File",
        command=self.simulate_input,
        font=("Arial", 14),
        bg="#7a7a52",
        fg="black"
        )
        self.simulate_button.pack(pady=5)
        self.simulate_button.place(x=70,y=430)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset = pd.read_csv(file_path, encoding='latin-1')
            self.dataset.drop_duplicates(keep='first', inplace=True)
            self.train_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Dataset Loaded Successfully!")

    def preprocessor(self, text):
        text = text.lower()
        text = re.sub(r"\\W", " ", text)
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
        message = self.test_entry.get("1.0", "end-1c")
        if message.strip():
            data = self.vectorizer.transform([message])
            prediction = self.clf.predict(data)[0]

            result_text = "! Spam" if prediction == 1 else "Not Spam"
            self.result_label.config(text=f"Prediction: {result_text}", fg="#ffffff")

            if prediction == 1:
                image_path = r"D:\\Saved Pictures\\spam image\\unsafe1.png"
            else:
                image_path = r"D:\\Saved Pictures\\spam image\\secure.png"

            image = Image.open(image_path).resize((100, 100), Image.LANCZOS)
            self.result_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.result_image)
            self.image_label.image = self.result_image

            self.history.append((message, result_text))


        else:
            messagebox.showwarning("Error", "Please enter a message to check.")

    def view_history(self):
        if not self.history:
            messagebox.showinfo("History", "No prediction history found.")
            return

        history_panel = tk.Toplevel(self.root)
        history_panel.title("Prediction History")
        history_panel.geometry("600x400")
        history_panel.configure(bg="#0099cc")

        tk.Label(history_panel, text="Prediction History", font=("Arial", 16, "bold"), bg="#0099cc", fg="white").pack(pady=10)

        frame = tk.Frame(history_panel)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(frame, wrap=tk.WORD, font=("Arial", 11),bg="Black", fg="white", yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        for idx, (msg, result) in enumerate(self.history, start=1):
            text_widget.insert(tk.END, f"{idx}. Message: {msg.strip()}\n   Prediction: {result}\n\n")

        text_widget.config(state=tk.DISABLED)

    def save_results(self):
        if not self.history:
            messagebox.showinfo("Info", "No results to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.DataFrame(self.history, columns=["Message", "Prediction"])
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Saved", f"Results saved to {file_path}")


    def simulate_input(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.test_entry.delete("1.0", tk.END)  # Clear old content
                    self.test_entry.insert(tk.END, content)  # Insert new content
                    messagebox.showinfo("Success", "Message loaded from file!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file:\n{e}")

    def clear_placeholder(self, event=None):
        if self.test_entry.get("1.0", "end-1c") == self.placeholder_text:
            self.test_entry.delete("1.0", tk.END)
            self.test_entry.config(fg="black")

    def add_placeholder_if_empty(self, event=None):
        if not self.test_entry.get("1.0", "end-1c").strip():
            self.test_entry.insert("1.0", self.placeholder_text)
            self.test_entry.config(fg="gray")
    

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()
