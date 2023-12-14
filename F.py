import torch
import tkinter as tk
from tkinter import filedialog, simpledialog
import seaborn as sns
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def get_user_input():
    root = tk.Tk()
    root.withdraw()
    
    user_input = [float(simpledialog.askstring("Input", f"Enter value {i + 1}")) for i in range(4)]
    root.destroy()
    
    return torch.tensor(user_input)

def load_model_ann(model_path):
    try:
        # load model
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_and_visualize(model, data):
    pickled_prediction = model.predict(np.array([data]))
    predict_label = np.argmax(pickled_prediction, axis=1)
    print("Predicted Label:", predict_label)

    # Visualization using Seaborn
    sns.barplot(x=list(range(len(pickled_prediction[0]))), y=pickled_prediction[0])
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.show()

if __name__ == '__main__':
    model_path = filedialog.askdirectory(title="Select Model Directory")
    model = load_model_ann(model_path)

    if model is not None:
        data = get_user_input()
        predict_and_visualize(model, data)
