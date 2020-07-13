# Pinterest dataset

URL: https://drive.google.com/file/d/0B0l8Lmmrs5A_REZXanM3dTN4Y28/view

[Pinterest](https://www.pinterest.com) is an online platform to create, manage and share boards with a collection of images, where each image is "pinned" to the board. Also, each board is tagged with a category. The data contains interactions ("pins"), as well as image and board category data.

### Files

* *subset_iccv_board_pins.bson*: contains pins associated to each board. In a recommender setting, each board can be considered as a user.
* *subset_iccv_pin_im.bson*:  contains image data for each pin. Multiple pins can refer to the same image. Each image includes a URL to retrieve the original image.
* *subset_iccv_board_cate.bson*: contains the "ground truth" of the boards' caregories.
* *categories.txt*: contains the index or id for each category in the previous file, according to the line number of each category in this file.

### Notes

* URLs for the original images could be disabled at any time.
* Dataset is really big: 882702 images in 46000 boards (created by 45580 unique users) and 2565241 interactions ("pins"). Also, each board is tagged with one of the 468 available categories.

### Reference

* X. Geng, H. Zhang, J. Bian and T. Chua, "Learning Image and User Features for Recommendation in Social Networks," 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, 2015, pp. 4274-4282.