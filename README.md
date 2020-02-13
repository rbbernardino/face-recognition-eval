# face-recognition-eval
Several face recognition algorithms evaluation for research purpose.

# tested models
## pyImageSearch code
- OpenCV + dlib

## pack2: Seetaface
### About
    - China National Computer vision group
    - CNN (convolutional neural net)
    - VIPLFaceNet (network)
      - 7 convolutional layers
      - 2 fully-connected layers
      - input size of 256x256x3
    - tailored from AlexNet to be fast and accurate
      - 40% more accurate
      - 80% faster training
      - 40% faster feature extraction 
    - based on cutting edge findings in deep learning research
    - 2016 last updated
    - pre-trained model
      - 1.4M face images
      - 16K subjects (Mongolians and Caucasians)
    - 2048-dimensional feature vector
    - on i7-3770 CPU: feature extraction in 115ms
### How to Run
1. **Detection**
   - go
