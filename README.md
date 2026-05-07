# Digit Sequence Reader

A model that reads a sequence of handwritten digits from a single stitched image and outputs the corresponding string of numbers. 

The model achieves this using an **Encoder-Decoder architecture with Bahdanau Attention**:
- The **Encoder** (CNN + Bidirectional LSTM) scans the image and extracts rich visual features across the entire sequence.
- The **Decoder** (LSTM) generates the predicted digits one by one. At each step, it uses the **Attention mechanism** to intelligently focus on the specific part of the image that corresponds to the digit it is currently predicting.

## Example Prediction

Below is an example of the model correctly predicting the sequence `8516313`.

**Sample Image:**
![Sample Image](img-readme/sample_5_8516313.png)

**Model Prediction & Attention Heatmap:**
![Prediction & Attention](img-readme/8516313.png)


