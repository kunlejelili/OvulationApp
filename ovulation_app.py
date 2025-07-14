#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import messagebox
from tkcalendar import Calendar
import joblib
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for file saving
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tkinter as tk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageTk
import sys

#  Utility functions
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

def get_resource_path(filename):
    """Get absolute path to resource (for PyInstaller and normal run)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

#  Main Application Class
class OvulationPredictor:
    def __init__(self, master):
        self.master = master
        master.title("Ovulation Predictor with Calendar")
       

        #  Input labels and entries
        labels = [
            'Age', 'Cycle Length', 'Luteal Phase', 'Avg BBT (°C)',
            'Mucus Score (1-5)', 'Mood Score (1-5)', 'Last Period Start Date (YYYY-MM-DD)'
        ]
        self.entries = []
        for label in labels:
            tk.Label(master, text=label).pack()
            entry = tk.Entry(master)
            entry.pack()
            self.entries.append(entry)

        #  Load logo (PNG only)
        logo_path = r"C:\Users\JK ADEDEJI\Desktop\OvulationApp\logo.png"
       
        try:
            img = Image.open(logo_path)
            self.logo_photo = ImageTk.PhotoImage(img)
            self.logo_label = tk.Label(master, image=self.logo_photo)
            self.logo_label.pack(pady=10)
        except Exception as e:
            print("Logo load failed:", e)

        # Buttons
        tk.Button(master, text="Predict & Show Calendar", command=self.predict).pack(pady=10)
        tk.Button(master, text="Reset Inputs", command=self.reset_inputs).pack(pady=5)
        tk.Button(master, text="📂 View Prediction History", command=self.show_history).pack(pady=5)
        tk.Button(master, text="📥 Export History to CSV", command=self.export_to_csv_desktop).pack(pady=5)
        tk.Button(master, text="📥 Export History to Excel", command=self.export_to_excel).pack(pady=5)
        tk.Button(master, text="📥 Export History to PDF", command=self.export_to_pdf).pack(pady=5)
        tk.Button(master, text="📈 Show Ovulation Trend Chart", command=self.plot_trend).pack(pady=5)

        # Calendar widget
        self.calendar = Calendar(master, selectmode='none')
        self.calendar.pack(pady=10)

        #  Load trained model
        self.model = joblib.load(get_resource_path("ovulation_model.pkl"))

    def predict(self):
        try:
            inputs = [float(entry.get()) for entry in self.entries[:-1]]
            last_period = self.entries[-1].get()
            start_date = datetime.strptime(last_period, "%Y-%m-%d")

            feature_names = ['age', 'cycle_length', 'luteal_phase', 'avg_bbt', 'mucus_score', 'mood_score']
            X_input = pd.DataFrame([inputs], columns=feature_names)
            predicted_day = round(self.model.predict(X_input)[0])

            ovulation_date = start_date + timedelta(days=predicted_day - 1)
            fertile_window = [ovulation_date + timedelta(days=i) for i in [-2, -1, 0, 1]]

            self.calendar.calevent_remove('all')
            self.calendar.selection_set(ovulation_date)
            for date in fertile_window:
                self.calendar.calevent_create(date, 'Fertile Window', 'fertile')
            self.calendar.tag_config('fertile', background='lightgreen', foreground='black')

            messagebox.showinfo("Prediction",
                f"Predicted Ovulation: {ovulation_date.date()}\n"
                f"Fertile Window: {fertile_window[0].date()} to {fertile_window[-1].date()}"
            )

            # Save prediction to CSV
            record = {
                'date_predicted': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'age': inputs[0],
                'cycle_length': inputs[1],
                'luteal_phase': inputs[2],
                'avg_bbt': inputs[3],
                'mucus_score': inputs[4],
                'mood_score': inputs[5],
                'last_period': last_period,
                'predicted_ovulation': ovulation_date.strftime('%Y-%m-%d'),
                'fertile_window_start': fertile_window[0].strftime('%Y-%m-%d'),
                'fertile_window_end': fertile_window[-1].strftime('%Y-%m-%d')
            }

            csv_file = "ovulation_prediction_history.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            else:
                df = pd.DataFrame([record])
            df.to_csv(csv_file, index=False)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_inputs(self):
        for entry in self.entries:
            entry.delete(0, tk.END)

    def show_history(self):
        csv_file = "ovulation_prediction_history.csv"
        if not os.path.exists(csv_file):
            messagebox.showinfo("No History", "No prediction history found yet.")
            return

        df = pd.read_csv(csv_file)
        history_window = tk.Toplevel(self.master)
        history_window.title("Prediction History")

        text = tk.Text(history_window, wrap="none", height=20, width=120)
        text.pack(expand=True, fill="both")
        text.insert("end", df.to_string(index=False))

    def export_to_csv_desktop(self):
        csv_file = "ovulation_prediction_history.csv"
        if not os.path.exists(csv_file):
            messagebox.showinfo("No History", "No prediction history found yet.")
            return

        df = pd.read_csv(csv_file)
        desktop = get_desktop_path()
        desktop_csv = os.path.join(desktop, "ovulation_prediction_history.csv")
        df.to_csv(desktop_csv, index=False)
        messagebox.showinfo("Export Successful", f"CSV copied to Desktop:\n{desktop_csv}")

    def export_to_excel(self):
        csv_file = "ovulation_prediction_history.csv"
        if not os.path.exists(csv_file):
            messagebox.showinfo("No History", "No prediction history found yet.")
            return

        df = pd.read_csv(csv_file)
        desktop = get_desktop_path()
        excel_file = os.path.join(desktop, "ovulation_prediction_history.xlsx")
        df.to_excel(excel_file, index=False)
        messagebox.showinfo("Export Successful", f"Excel file saved to:\n{excel_file}")

    def export_to_pdf(self):
        csv_file = "ovulation_prediction_history.csv"
        if not os.path.exists(csv_file):
            messagebox.showinfo("No History", "No prediction history found yet.")
            return

        df = pd.read_csv(csv_file)
        desktop = get_desktop_path()
        pdf_file = os.path.join(desktop, "ovulation_prediction_history.pdf")

        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica", 10)

        margin = 40
        y = height - margin
        line_height = 14

        columns = df.columns.tolist()
        header = ' | '.join(columns)
        c.drawString(margin, y, header)
        y -= line_height

        for _, row in df.iterrows():
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin
                c.drawString(margin, y, header)
                y -= line_height

            row_text = ' | '.join(str(row[col]) for col in columns)
            c.drawString(margin, y, row_text)
            y -= line_height

        c.save()
        messagebox.showinfo("Export Successful", f"PDF saved to:\n{pdf_file}")

    def plot_trend(self):
        try:
            csv_file = "ovulation_prediction_history.csv"
            if not os.path.exists(csv_file):
                messagebox.showinfo("No History", "No prediction history found yet.")
                return

            df = pd.read_csv(csv_file)
            df['date_predicted'] = pd.to_datetime(df['date_predicted'])
            df['predicted_ovulation_day'] = pd.to_datetime(df['predicted_ovulation'])

            plt.figure(figsize=(10, 5))
            plt.plot(df['date_predicted'], df['predicted_ovulation_day'], marker='o', linestyle='-')
            plt.title("Ovulation Prediction Trend Over Time")
            plt.xlabel("Date Predicted")
            plt.ylabel("Predicted Ovulation Date")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save chart to PDF
            desktop = get_desktop_path()
            chart_pdf = os.path.join(desktop, "ovulation_trend_chart.pdf")
            try:
                plt.savefig(chart_pdf)
                plt.close()
                messagebox.showinfo("Chart Saved", f"Trend chart saved to:\n{chart_pdf}")
            except Exception as e:
                messagebox.showerror("Error Saving Chart", str(e))
        except Exception as e:
            messagebox.showerror("Trend Plot Error", str(e))

#  Launch the App
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Ovulation Predictor")
    root.iconbitmap(r"C:\Users\JK ADEDEJI\Desktop\OvulationApp\icon.ico")
    app = OvulationPredictor(root)
    root.mainloop()


# In[ ]:





# In[ ]:




