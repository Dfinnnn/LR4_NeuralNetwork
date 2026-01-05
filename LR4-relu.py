import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

st.set_page_config(page_title="ReLU Activation Function", layout="centered")

st.title("ReLU Activation Function Visualisation")
st.write("### f(x) = max(0, x)")

# Slider for x-range
x_min, x_max = st.slider("Select x-range", -10.0, 10.0, (-6.0, 6.0))

# Create tensor input
x = torch.linspace(x_min, x_max, 400)
relu = nn.ReLU()
y = relu(x)

# Plot
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.axhline(0)
ax.axvline(0)
ax.set_xlabel("Input x")
ax.set_ylabel("Output f(x)")
ax.set_title("ReLU Activation Function")

st.pyplot(fig)

st.write("""
**Analysis**

ReLU outputs:
- 0 when x < 0  
- x when x ≥ 0  

Advantages:
- reduces vanishing gradient
- easy to compute
- speeds up deep learning convergence
""")
