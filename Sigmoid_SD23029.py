import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Page Configuration
st.set_page_config(page_title="Sigmoid Visualization", layout="centered")

st.title("2. Sigmoid Activation Function")
st.markdown("### Formula: $f(x) = \\frac{1}{1 + e^{-x}}$")

# Sidebar Control
st.sidebar.header("Settings")
x_range = st.sidebar.slider("Select range for X", 5.0, 20.0, 10.0)

# Data Generation
x = torch.linspace(-x_range, x_range, 400)
sigmoid = nn.Sigmoid()
y = sigmoid(x)

# Plotting
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy(), color='green', linewidth=2, label="Sigmoid Output")
ax.axhline(0, color='black', linewidth=0.5, linestyle="--")
ax.axhline(1, color='black', linewidth=0.5, linestyle="--")
ax.axvline(0, color='black', linewidth=0.5, linestyle="--")
ax.set_title("Sigmoid Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.write("""
**Observation:**
Sigmoid 'squashes' the input values into a range between **0 and 1**. 
It is often used for binary classification probabilities but can suffer from vanishing gradients at extreme values.
""")