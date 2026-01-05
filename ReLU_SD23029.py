import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Page Configuration
st.set_page_config(page_title="ReLU Visualization", layout="centered")

st.title("1. ReLU Activation Function")
st.markdown("### Formula: $f(x) = \max(0, x)$")

# Sidebar Control
st.sidebar.header("Settings")
x_range = st.sidebar.slider("Select range for X", 5.0, 20.0, 10.0)

# Data Generation
x = torch.linspace(-x_range, x_range, 400)
relu = nn.ReLU()
y = relu(x)

# Plotting
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy(), color='blue', linewidth=2, label="ReLU Output")
ax.axhline(0, color='black', linewidth=0.5, linestyle="--")
ax.axvline(0, color='black', linewidth=0.5, linestyle="--")
ax.set_title("Rectified Linear Unit (ReLU)")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.write("""
**Observation:**
ReLU outputs **0** for all negative inputs and **x** for positive inputs. 
This linearity for positive values helps solve the vanishing gradient problem in deep neural networks.
""")