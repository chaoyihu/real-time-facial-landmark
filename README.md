# real-time-facial-landmark

## Abstract
Synthetic human facial datasets offer many advantages such as controllable production and elimination of ethical concerns. In this project, we developed a facial landmarks model that trains on synthetic facial data and generalizes to real facial images.

The network architecture we adopted consists of 68 layers featuring a Resnet backbone. For the selection of loss function, we conducted a pair of experiments showing that adopting wing loss reduced the prediction mean error (ME) and improved the prediction success rate (SR) by 7.8% compared to using MSE loss.

We experimented with different portions of synthetic data in the training set. Trained with 100% synthetic data and tested on real images, the model achieved ME ± Std = 23.69 ± 8.71 and SR = 0.41 (ME threshold = 20.0). When trained with 60% synthetic data and 40% real data, the performance improved to ME ± Std = 18.64 ± 7.88 and SR = 0.60. However, decreasing the percentage of real data to 10% was shown to be detrimental to prediction accuracy.

To evaluate the potential of the model inferencing on consumer devices in real time, we took special notes to the model size and inference speed. Experimenting with a trimmed network 23.6% smaller in size compared to the full version, we found that it inferences 23.1% faster at the cost of a 29.0% drop in SR, demonstrating a trade-off between model size and performance.

Integrating the landmarks predictor with a face detector, we delivered a desktop demo that runs landmarks inferences in real-time on video inputs at approximately 20 FPS.

## Detailed Presentation

![Slide1](doc/readme_slides/slides1.jpg)
![Slide2](doc/readme_slides/slides2.jpg)
![Slide3](doc/readme_slides/slides3.jpg)
![Slide4](doc/readme_slides/slides4.jpg)
![Slide5](doc/readme_slides/slides5.jpg)
![Slide6](doc/readme_slides/slides6.jpg)
![Slide7](doc/readme_slides/slides7.jpg)
![Slide8](doc/readme_slides/slides8.jpg)
![Slide9](doc/readme_slides/slides9.jpg)
![Slide10](doc/readme_slides/slides10.jpg)
<img src="doc/readme_slides/slides10_gif1.gif" width="400" />
<img src="doc/readme_slides/slides10_gif2.gif" width="400" />
![Slide11](doc/readme_slides/slides11.jpg)
![Slide12](doc/readme_slides/slides12.jpg)
![Slide13](doc/readme_slides/slides13.jpg)
![Slide14](doc/readme_slides/slides14.jpg)
![Slide15](doc/readme_slides/slides15.jpg)
![Slide16](doc/readme_slides/slides16.jpg)
![Slide17](doc/readme_slides/slides17.jpg)
![Slide18](doc/readme_slides/slides18.jpg)
![Slide19](doc/readme_slides/slides19.jpg)
![Slide20](doc/readme_slides/slides20.jpg)

## Conclusion and Prospects
In this project, we conducted experiments to train a facial landmarks model using synthetic facial data and evaluate its ability to generalize to real facial images. We employed a lightweight convolutional network architecture featuring ResNet backbone and wing loss for training. The choice of loss function was made based on experiment results showing that adoption of wing loss improved the prediction SR by 7.8%. Our results suggest that trained with 100% synthetic data, the model achieved ME ± Std = 23.69 ± 8.71 and SR = 0.41 (ME threshold = 20.0) on real test images. The performance improved to ME ± Std = 18.64 ± 7.88 and SR = 0.60 when trained with 60% synthetic data and 40% real data, and dropped significantly when the percentage of real data was further decreased to 10%. The results indicates that for the data mixing approach, a proper mixture of synthetic and real data in the training set might help improve the model performance compared to training with purely synthetic data. We further explored the impact of model size on its performance. A trimmed network 23.6% smaller in size compared to the full version inferences 23.1% faster at the cost of a 29.0% drop in SR. Results suggest a trade-off relationship between model inference speed and its prediction accuracy. Employing the landmarks model trained in this project integrated with a facial detection module, we developed a desktop demo that runs landmarks inferences in real-time on webcam capture or video inputs at approximately 20 FPS.

Overall, we have fulfilled our project design in the project proposal with additional experiments on data mixing in the training set and comparison between MSE and wing loss functions. The prediction accuracy of our models cannot match state-of-the-art landmarks localization models, yet our results did provide some revelations subject to further exploration.
