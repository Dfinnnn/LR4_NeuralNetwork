import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation", layout="centered")

st.title("ReLU Activation Function")
st.write("""
**Rectified Linear Unit (ReLU)** is defined as:

f(x) = max(0, x)

It outputs zero for negative inputs and a linear value for positive inputs.
""")

# Input range
x_min = st.slider("Minimum x value", -10.0, 0.0, -5.0)
x_max = st.slider("Maximum x value", 0.0, 10.0, 5.0)

x = np.linspace(x_min, x_max, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("ReLU Activation Function")
ax.grid(True)

st.pyplot(fig)
