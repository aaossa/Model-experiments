# UGallery dataset

URL: https://drive.google.com/drive/folders/1Dk7_BRNtN_IL8r64xAo6GdOYEycivtLy

[UGallery](https://www.ugallery.com) is an online gallery that connects artists and collectors. The providad data contains transactions (hashed user and item IDs due to privacy requirements) as well as visual embeddings of the painting images files.

### Files

* *ugallery_inventory.csv*: contains tuples in the form of `(artwork_id_hash, artist_id_hash, upload_timestamp)`. Each tuple has the time an item was added to the website inventory.

* *ugallery_purchases.csv*: contains tuples in the form of `(user_id_hash, purchase_timestamp, artwork_id_hash)`. Requires grouping by user and timestamp to retrieve baskets (see [Notes](#Notes)).
* *ugallery_resnet50_embeddings.npy*: Numpy array of shape (13297, 2), where each row is of shape (2,) where the first value is the `artwork_id_hash`, and the second one is the ResNet50 embeddings of the artwork image.

### Notes

* Most of the paintings (78%) are one-of-a-king, i.e., the is a single instance of each item and once is purchased, is removed from the inventory. Also, most purchase transactions have been made over these items (81.7%).
* Because these are physical artworks, the availability of the items must be simulated in order to make recommendations. 
* Each item has a single creator (artist). In this dataset there are 573 artists, who have uploaded 10.54 items in average to the online art store.
* In the reference paper, purchases are grouped into baskets containing one or more artworks (same timestamp means same purchase).
* Last purchase/basket was used as test data for each user.
* Transactions data contains 6535 transactions of 2919 users on 6030 items.

### Reference

* CuratorNet: A Neural Network for Visually-aware Recommendation of Art Images