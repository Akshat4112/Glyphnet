# GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks
Akshat Gupta, Laxman Singh Tomar, Ridhima Garg <br>
[Website](https://akshat4112.github.io/Glyphnet/) | [Paper](https://akshat4112.github.io/Glyphnet/) <br>

**Introduction**<br>
In cyber security, attackers employ different attacks to infiltrate our systems and networks, with the objective varying from stealing crucial information to inflicting system damage. One such deceptive attack is the homoglyph attack (Woodbridge et al. 2018), which involves an attacker trying to fool humans and computer systems by using characters and symbols that may appear visually similar to characters used in real domain and process names but  are different. For example, a typical homoglyph attack may involve changing “d” to “cl”, “o” to “θ”, and “l” to “1”. <br>

![Real and Homoglyph domains](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/real_fake_domains.png) <br>

![Real Rober Frost](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/realfrost.png) <br>
![Fake Robert Frost](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/fakefrost.png) <br>
**Dataset** <br>

![Real and Homoglyph Dataset](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/real-homoglyph.png) <br>
***Proposed Dataset***

We have proposed a dataset consisting of real and homoglyph domains. In order to generate homoglyph domains,real domains are needed. We have obtained domains from the Domains Project(Turkynewych 2020). This repository is one of the largest collections of publicly available active domains. The entire repository comprises 500M domains, and we restricted our work to 2M domains due to hardware restrictions.

**Methodology**<br>

![Model Architecture](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/architecture.png) <br>
![Attention Layer](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/attention_layer.png) <br>




