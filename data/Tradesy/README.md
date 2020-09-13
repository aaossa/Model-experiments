# Tradesy dataset

URL: http://jmcauley.ucsd.edu/data/tradesy/

[Tradsy](https://www.tradesy.com/) is an online platform to trade second-hand clothing inside its community. Each user has four lists: "seeling", "sold", "bought", and "want", but only "bought" and "want" will be considered as possitive feedback. The data contains interactions, as well as visual features extracted using AlexNet for each image.

### Files

* *tradesy.json.gz*: contains items for the different user lists ("selling", "sold", "bought", and "want") for every user.
* *image_features_tradesy.b*:  contains image visual features (a vector of 4096 floats) extracted using a Caffe implementation of AlexNet.

### Notes

* URLs for the original images can also be found in the dataset URL, but could be disabled at any time.
* Dataset is really big: 166526 images, 19823 users, and 410186 interactions, and this is considering only two types of feedback.

### Reference

* R. He, J. McAuley, "VBPR: Visual bayesian personalized ranking from implicit feedback", *AAAI*, 2016