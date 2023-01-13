# How to Download KITTI and VKITTI

## VKITTI: 

Refer to their [homepage](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/). Copy the link to their RGB dataset and call wget. 

```
wget https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
```

Unzip the tar file with

```
tar -xvf vkitti_2.0.3_rgb.tar
```

## KITTI:

Refer to the [Object Tracking Evaluation page](https://www.cvlibs.net/datasets/kitti/eval_tracking.php). Call wget on the left color images. 

```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
```

Unzip the zip file with 

```
unzip data_tracking_image_2.zip
```