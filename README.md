## Part-of-Speech Tagger for Twitter Data
### Introduction
This project develops a Part-of-Speech (POS) tagger using Hidden Markov Models (HMM) tailored for analyzing Twitter tweets. The system progresses from basic POS tagging using naive methods to more advanced implementations leveraging the Viterbi algorithm for improved accuracy.

### Installation
To set up this project, ensure you have Python 3.x installed along with the required libraries: pandas, numpy, and re. You can install the dependencies using pip:


`pip install pandas numpy`

### Usage
The project is structured into various components, each corresponding to a specific task outlined in the project guidelines.

Basic Probability Calculations: Run the q2a() function to compute initial token-tag probabilities and output them to output_probs.txt.
Naive Prediction: Execute the naive_predict() function for a baseline tagging prediction based on the highest probability tag for each token.
Viterbi Algorithm Implementation: Use the viterbi_predict() function to apply the Viterbi algorithm for POS tagging, utilizing transition and emission probabilities for improved accuracy.
### Example
To run the basic probability calculation and generate the initial output probabilities, use the following command:


`from hmm import q2a` <br>
`q2a()`
### Acknowledgments
This project was developed based on the guidelines provided by the professors for the BT3102 module, with tasks ranging from basic POS tagging to implementing and refining the Viterbi algorithm for enhanced performance on Twitter data.
