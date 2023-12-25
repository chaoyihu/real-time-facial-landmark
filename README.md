# real-time-facial-landmark

## Abstract
Synthetic human facial datasets offer many advantages such as controllable production and elimination of ethical concerns. In this project, we developed a facial landmarks model that trains on synthetic facial data and generalizes to real facial images.

The network architecture we adopted consists of 68 layers featuring a Resnet backbone. For the selection of loss function, we conducted a pair of experiments showing that adopting wing loss reduced the prediction mean error (ME) and improved the prediction success rate (SR) by 7.8% compared to using MSE loss.

We experimented with different portions of synthetic data in the training set. Trained with 100% synthetic data and tested on real images, the model achieved ME ± Std = 23.69 ± 8.71 and SR = 0.41 (ME threshold = 20.0). When trained with 60% synthetic data and 40% real data, the performance improved to ME ± Std = 18.64 ± 7.88 and SR = 0.60. However, decreasing the percentage of real data to 10% was shown to be detrimental to prediction accuracy.

To evaluate the potential of the model inferencing on consumer devices in real time, we took special notes to the model size and inference speed. Experimenting with a trimmed network 23.6% smaller in size compared to the full version, we found that it inferences 23.1% faster at the cost of a 29.0% drop in SR, demonstrating a trade-off between model size and performance.

Integrating the landmarks predictor with a face detector, we delivered a desktop demo that runs landmarks inferences in real-time on video inputs at approximately 20 FPS.

## Detailed Presentation

![Slide1](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/580b55df-c3ac-49e4-9b3d-454cfbb56e34)
![Slide2](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/189acd63-56bf-44d9-9d36-8682878a6dfa)
![Slide3](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/a50ffd18-799e-49e3-8403-2d0e69c3c680)
![Slide4](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/440b796f-aa25-427f-9a32-b944a36f5003)
![Slide5](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/4dffcf6b-46ab-4a76-a7e1-8794500e535c)
![Slide6](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/99d8af54-e891-4ba6-a3fe-e4a04b8e8dc0)
![Slide7](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/11523b6b-f13f-4028-ba1a-ac7a64920e1d)
![Slide8](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/db0770a0-6074-4ad8-be7a-ff2cf377f0ba)
![Slide9](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/5568a309-e78d-46f5-8c75-1c0015c5e710)
![Slide10](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/ace13c9a-74c7-4fc1-8344-a248682d21b2)
<img src="https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/3c83a23b-280b-453e-8811-e4dddd2e8e7f" width="400" />
<img src="https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/957699ae-5fe5-4e58-92a8-6de82ab883ea" width="400" />
![Slide11](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/52647448-3e88-4e37-8559-6dfb503c1c80)
![Slide12](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/adc0fc9a-9d40-4bb2-9674-b2a2d5b8afee)
![Slide13](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/0d2c5797-9986-430d-8624-04559b544fbf)
![Slide14](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/c7261885-d9b6-40a5-8bf1-dc826019d6e0)
![Slide15](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/a0e1acdc-e55c-4e64-925b-b80b00066398)
![Slide16](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/99c4f7cc-c3f5-4b4a-87a6-a0718c61ad90)
![Slide17](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/05ff6bc1-a7c9-4d0a-95f5-53d7aae7c396)
![Slide18](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/3d0ff77d-fdec-4d0d-9303-b82cf2041b34)
![Slide19](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/9f53a70d-1310-4f35-a08f-db5bbdbece44)
![Slide20](https://github.com/zoehcycy/real-time-facial-landmark/assets/46927076/c785b019-9368-4949-a6a9-2aaba33edf83)

## Conclusion and Prospects
In this project, we conducted experiments to train a facial landmarks model using synthetic facial data and evaluate its ability to generalize to real facial images. We employed a lightweight convolutional network architecture featuring ResNet backbone and wing loss for training. The choice of loss function was made based on experiment results showing that adoption of wing loss improved the prediction SR by 7.8%. Our results suggest that trained with 100% synthetic data, the model achieved ME ± Std = 23.69 ± 8.71 and SR = 0.41 (ME threshold = 20.0) on real test images. The performance improved to ME ± Std = 18.64 ± 7.88 and SR = 0.60 when trained with 60% synthetic data and 40% real data, and dropped significantly when the percentage of real data was further decreased to 10%. The results indicates that for the data mixing approach, a proper mixture of synthetic and real data in the training set might help improve the model performance compared to training with purely synthetic data. We further explored the impact of model size on its performance. A trimmed network 23.6% smaller in size compared to the full version inferences 23.1% faster at the cost of a 29.0% drop in SR. Results suggest a trade-off relationship between model inference speed and its prediction accuracy. Employing the landmarks model trained in this project integrated with a facial detection module, we developed a desktop demo that runs landmarks inferences in real-time on webcam capture or video inputs at approximately 20 FPS.

Overall, we have fulfilled our project design in the project proposal with additional experiments on data mixing in the training set and comparison between MSE and wing loss functions. The prediction accuracy of our models cannot match state-of-the-art landmarks localization models, yet our results did provide some revelations subject to further exploration.
