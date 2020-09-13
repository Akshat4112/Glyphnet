# from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import streamlit as st

st.title("Phishing attack prediction based on Homoglyphs")
domain_name  = st.text_input("Enter the domain name...")
if st.button("Predict"):
    result = domain_name.title()
    st.success(str.lower(result))
