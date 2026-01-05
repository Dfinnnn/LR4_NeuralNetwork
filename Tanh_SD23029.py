import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Page Configuration
st.set_page_config(page_title="Tanh Visualization", layout="centered")

st.title("3. Tanh Activation Function")
st.markdown("### Formula: $f(x) = \\tanh(x)$")

# Sidebar Control
st.sidebar.header("Settings")
x_range = st.sidebar.slider("Select range for X", 5.0, 20.0, 10.0)

# Data Generation
x = torch.linspace(-x_range, x_range, 400)
tanh = nn.Tanh()
y = tanh(x)

# Plotting
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy(), color='red', linewidth=2, label="Tanh Output")
ax.axhline(0, color='black', linewidth=0.5, linestyle="--")
ax.axhline(1, color='gray', linewidth=0.5, linestyle="--")
ax.axhline(-1, color='gray', linewidth=0.5, linestyle="--")
ax.axvline(0, color='black', linewidth=0.5, linestyle="--")
ax.set_title("Hyperbolic Tangent (Tanh)")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.write("""
**Observation:**
Tanh is similar to Sigmoid but maps inputs to a range between **-1 and 1**.
Because it is zero-centered, it is often preferred over Sigmoid for hidden layers in neural networks.
""")