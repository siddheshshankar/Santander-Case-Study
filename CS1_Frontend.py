import time
import pickle
import lightgbm
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# Streamlit Frontend
st.markdown("<h2 style='text-align: center;'><u>Santander Customer Transaction Prediction</u></h2>", unsafe_allow_html=True)
st.image("https://www.signotec.com/medien/bilder/2018_09_10_santander_logo__1__.png", width=None)
st.write("Santander Bank, N. A., formerly Sovereign Bank, is a wholly owned subsidiary of the Spanish Santander Group. It is based in Boston and its principal market is the northeastern United States. Sovereign Bank was rebranded as Santander Bank on October 17, 2013. In 2018, Santander bank hosted a competition on kaggle to predict whether a particular customer with given details would transact or not. Below are the kaggle scores and model performace scores:")
col1, col2, col3, col4, col5 = st.beta_columns([2.9, 1.9, 1.65, 2.4, 2])

with col1:
	st.success("Kaggle score: 90%")
with col2:
	st.success("Rank: 374")
with col3:
	st.success("Top 4%")
with col4:
	st.success("Accuracy 98%")
with col5:
	st.success("F1 90%")

st.write("Please provide a text file with all the features in order to predict which customer will make a specific transaction in the future, irrespective of the amount of money transacted.")


# Load file
uploaded_file = st.file_uploader("",type='txt')

# Light GBM Model
def run_model(uploaded_file):
	"""
	Main function to run ML model
	"""
	try:
		start_time = time.time()
		store_result = []
		model = pickle.load(open("finalized_model.pkl", 'rb'))
		file = pd.read_csv(uploaded_file, sep='\t')
		autoencoder = load_model("autoencoder_model.h5")

		for i in range(len(file)):
			extra_features = autoencoder.predict(file.iloc[i, 1:].values.astype('float32').reshape(1, 200))
			total_features = np.append(file.iloc[i, 1:].values.astype('float32').reshape(1, 200), extra_features).reshape(1, -1)
			if model.predict(total_features)[0] == 1:
				store_result.append('Transacted')
			else:
				store_result.append('No Transaction')
	except ValueError:
		st.warning('Upload a text file.')

	st.dataframe(pd.DataFrame({'ID Code': file.iloc[:, 0], 'Prediction': store_result}).assign(hack='').set_index('hack'))
	st.success(f'Execution time: {round(time.time() - start_time, 2)} seconds')

# Run Model
if st.button('Predict'):
	run_model(uploaded_file)
	

# streamlit run CS1_Frontend.py
