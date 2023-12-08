
# GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks
GlyphNet is a project aimed at detecting homoglyph attacks using an attention-based Convolutional Neural Network (CNN). It leverages a unique approach involving image datasets of domain names, real and homoglyphs, to train the model.
[Website](https://akshat4112.github.io/Glyphnet/) | [Paper](https://akshat4112.github.io/Glyphnet/) <br>

![Real and Homoglyph domains](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/real_fake_domains.png) <br>

## Contents
- `data/`: Contains the image dataset used in the project.
- `notebooks/`: Jupyter Notebooks illustrating the methodology and experiments.
- `src/`: Source code for the CNN model and data processing.

**Dataset** <br>

![Real and Homoglyph Dataset](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/real-homoglyph.png) <br>
***Proposed Dataset***

We have proposed a dataset consisting of real and homoglyph domains. In order to generate homoglyph domains,real domains are needed. We have obtained domains from the Domains Project(Turkynewych 2020). This repository is one of the largest collections of publicly available active domains. The entire repository comprises 500M domains, and we restricted our work to 2M domains due to hardware restrictions.

**Methodology**<br>

![Model Architecture](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/architecture.png) <br>
![Attention Layer](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/attention_layer.png) <br>

## Getting Started
- Follow installation instructions in `requirements.txt`.
- Refer to Jupyter Notebooks for a detailed walkthrough.

## Contribution
Contributions to enhance detection methods and dataset quality are welcome. Please adhere to contribution guidelines.

## License
This project is under the MIT License.


