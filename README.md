# Phishing Detection using Convnets

Cyber attacks are meant to deceive machines in believing something which doesn’t exist at the first place. But there are some for which even humans fall prey to. One such famous attack that has been used by attackers over the years to exploit the vulnerability of vision is known to be as **Homoglyph attack**. It employs a simple yet effective mechanism of creating such illegitimate domains which are hard to be differentiated from the legit ones. And as the difference is pretty indistinguishable for a user to notice, they just cannot stop themselves from clicking on these spoof domain names. In many cases, which results into either information theft or malware attack at their systems.

To address the aforementioned problem,we introduced an unpaired homoglyph attack detection system using a *Convolutional Neural Network*. Our model achieves state-of-the-art in detecting in domain spoof homoglyphs and process spoof homoglyphs showing 0.93 AUC and 0.98 AUC respectively.

All data files under /data
All notebook files under /notebooks
All script files under /src/scripts
All model files under models/
All visualization under /visualizations

### Process to run 

1. Run get_data.py to download data
2. Run data_preprocess.py to create images from data
3. Run train_cnn.py to train a Convolutional NN, model is saved in /models
4. Run predict_cnn.py to make inference to a model
5. Run app.py to run web app for the same.
