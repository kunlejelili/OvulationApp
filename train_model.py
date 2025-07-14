#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the synthetic data
df = pd.read_csv("synthetic_ovulation_data.csv")

# Features and target
X = df[['age', 'cycle_length', 'luteal_phase', 'avg_bbt', 'mucus_score', 'mood_score']]
y = df['ovulation_day']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model trained! MAE: {mae:.2f}")

# Save model
import joblib
joblib.dump(model, 'ovulation_model.pkl')


# In[ ]:





# In[ ]:




