# GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks

Akshat Gupta, Laxman Singh Tomar, Ridhima Garg

**Abstract**
Cyber attacks deceive machines into believing something that does not exist in the first place. However, there are some to which even humans fall prey. One such famous attack that attackers have used over the years to exploit the vulnerability of vision is known to be a Homoglyph attack. It employs a primary yet effective mechanism to create illegitimate domains that are hard to differentiate from legit ones. Moreover, as the difference is pretty indistinguishable for a user to notice, they cannot stop themselves from clicking on these homoglyph domain names.
In many cases, that results in either information theft or malware attack on their systems. Existing approaches use simple, string-based comparison techniques applied in primary language-based tasks. Although they are impactful to some extent, they usually fail because they are not robust to different types of homoglyphs and are computationally not feasible because of their time requirement proportional to the string's length.
Similarly, neural network-based approaches are employed to determine real domain strings from fake ones. Nevertheless, the problem with both methods is that they require paired sequences of real and fake domain strings to work with, which is often not the case in the real world, as the attacker only sends the illegitimate or homoglyph domain to the vulnerable user. Therefore, existing approaches are not suitable for practical scenarios in the real world. In our work, we created GlyphNet, an image dataset that contains 4M domains, both real and homoglyphs. Additionally, we introduce a baseline method for homoglyph attack detection system using an attention-based convolutional Neural Network. We show that our model can reach state-of-the-art accuracy in detecting homoglyph attacks with a 0.93 AUC on our dataset. 

**Introduction**<br>
In cyber security, attackers employ different attacks to infiltrate our systems and networks, with the objective varying from stealing crucial information to inflicting system damage. One such deceptive attack is the homoglyph attack (Woodbridge et al. 2018), which involves an attacker trying to fool humans and computer systems by using characters and symbols that may appear visually similar to characters used in real domain and process names but  are different. For example, a typical homoglyph attack may involve changing “d” to “cl”, “o” to “θ”, and “l” to “1”. <br>

![Real and Homoglyph domains](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/real_fake_domains.png) <br>

![Real Rober Frost](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/realfrost.png) <br>
![Fake Robert Frost](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/fakefrost.png) <br>
**Dataset** <br>

![Real and Homoglyph Dataset](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/real-homoglyph.png) <br>
***Proposed Dataset***

We have proposed a dataset consisting of real and homoglyph domains. In order to generate homoglyph domains,real domains are needed. We have obtained domains from the Domains Project(Turkynewych 2020). This repository is one of the largest collections of publicly available active domains. The entire repository comprises 500M domains, and we restricted our work to 2M domains due to hardware restrictions.

***Homoglyph Creation Algorithm***

Homoglyph Generation is an important task, as one needs to ensure enough randomness to make it appear real and keep the process simple enough to fool the target. Publicly available tools like dnstwist(Ulikowski 2015) replace every character in the real input domain with their respective glyphs. It generates poor homoglyphs for the large part because it relies on paired data which is not fit to serve the purpose practically. We created our novel algorithm for the generation of homoglyph domains to ensure that real homoglyphs are generated with randomness and closeness. To achieve this, we sample homoglyph noise characters using Gaussian sampling(Boor, Overmars, and Van Der Stappen 1999) from the glyph pool. We used 1M real domains to generate 2M homoglyphs with a single glyph character and introduce diversity in our dataset; we reran this algorithm on the remaining 1M real domains to generate homoglyph domains with two character glyphs. Finally, we have the 4M real and homoglyph domains.

***Image Generation***
Homoglyph attacks exploit the weakness of human vision to differentiate real from homoglyph domain names. From a visual perspective, we are interested in learning the visual characteristics of real and homoglyph domain names. To do so, we rendered images from the real and homoglyph strings generated via our algorithm. We have used ARIAL Typeface as our chosen font, 28 font size, on a black background with white text from the middle left of the image; the image size is 150 × 150.

**Methodology**<br>

![Model Architecture](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/architecture.png) <br>


![Attention Layer](https://github.com/Akshat4112/Glyphnet/blob/pages/resources/attention_layer.png) <br>



**Experimentation**<br>
**Results**<br>
**References**<br>
**Citation**<br>





