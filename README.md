# Phishing Detection using Convnets

Cyber attacks are meant to deceive machines in believing something which doesn’t exist at the first place. But there are some for which even humans fall prey to. One such famous attack that has been used by attackers over the years to exploit the vulnerability of vision is known to be as **Homoglyph attack**. It employs a simple yet effective mechanism of creating such illegitimate domains which are hard to be differentiated from the legit ones. And as the difference is pretty indistinguishable for a user to notice, they just cannot stop themselves from clicking on these spoof domain names. In many cases, which results into either information theft or malware attack at their systems.

To address the aforementioned problem,we introduced an unpaired homoglyph attack detection system using a *Convolutional Neural Network*. Our model achieves state-of-the-art in detecting in domain spoof homoglyphs and process spoof homoglyphs showing 0.93 AUC and 0.98 AUC respectively.

All data files under /data
All notebook files under /notebooks
All script files under /code
All model files under models/
All visualization under /visualizations

## Process to run 

### Data generation
1. Run dataGeneration.py to prepare the data which will create a CSV. It accepts an path argument for data.
It uses domains_final.txt which is already present in the data folder.

   - ``python code/dataGeneration.py --path_data <path_for_data>``

   
2. Run ImageGeneration.py to create images from CSV file created at step 1. It will create directories of real and fake in the dataset folder.

    - ``python code/ImageGeneration.py --path_data <path_for_data>``
   

3. Run dataSplit.py to to split the data into train, val, test.

    - ``python code/dataSplit.py --path_data <path_for_data>``

Note -: You can also directly use th dataset present in our assets<link> if you dont want to generate from scratch.

### Model



### Training the model

- Run the train.py




