# SurgT_labelling

[![GitHub stars](https://img.shields.io/github/stars/Cartucho/SurgT_labelling.svg?style=social&label=Stars)](https://github.com/Cartucho/SurgT_labelling)

Goal: Get ground truth bounding boxes given a stereo video + camera calibration parameters.

<img src="https://user-images.githubusercontent.com/15831541/152753224-e5e87ab7-e508-4aba-92ae-3e6a0587b246.png">

<img src="https://user-images.githubusercontent.com/15831541/152757790-30a8658b-e7cb-44f9-93ef-811401f90e30.png">

On the bottom-left of the interface you can see the ids of the current image pair `Im:` and bounding box `Id:`. You can change the keys according to your preferance by editing the [config.yaml](https://github.com/Cartucho/stereo_labeling/blob/main/config.yaml) file. It is also in [config.yaml](https://github.com/Cartucho/stereo_labeling/blob/main/config.yaml) where you set the input directory, video and calibration parameters.

## Labelling instructions for the SurgT MICCAI challenge

You will be labelling the bounding boxes by clicking on the same target in both images. You want to guarantee that the **centre** of the bounding box is always targeting the same tissue region.
For the SurgT challenge, the annotators are never allowed to use `i`nterpolation, automation or external intervention to aid their labelling!

#### Definitions

For each target bounding box:

- Set `is_visible_in_both_stereo = False` if the target's centre point is **(1) fully out-of-view** or **(2) fully occluded** in the left or in the right image. In other words, when the target's centre is compelety NOT visible in either of the stereo images. Otherwise `is_visible_in_both_stereo = True`.

- Set `is_difficult = True` if the target's centre **(1) annotation is too difficult**, or there are **(2) conflicting opinions between annotators**, or **(3) there is fast motion of the camera or the tissue**. The frames that are marked as difficult (`is_difficult = True`) do not affect any of the metrics/scores. In practice, in our benchmarking tool, the difficult frames are simply ignored. The reason for marking frame-pairs as difficult during (3) fast motion of the camera or tissue, is that it is possible to have error in the stereo-camera's synchronization. Therefore, the left and right image may be captured at slightly different timestamps, which is a problem during fast-camera-motion since the target will no longer be imaged consistently between the stereo images. For example the target may no longer be row-aligned as it is expected in rectified images. The faster the motion, the easier it is to notice synchronization errors. By labeling these image-pairs as `is_difficult = True` they are correctly ignored by the benchmarking tool.

#### How to label?

The usage idea is the following:
1. The annotator should watch the entire video and decide which keypoint will be labelled next. We recommend the annotator to choose a keypoint that is easy to label throughout the video;
2. Then, the annotator should classify `is_visible_in_both_stereo` for all the frame-pairs of the video. This can be done by pressing `v` over an image, or `v` over a range of images. A big red `X` will be draw over the images with `is_visible_in_both_stereo = False`;
3. Then the annotator should labell the keypoint in all the remaining images, where `is_visible_in_both_stereo = True`. All the keypoints should respect:
    1. The annotator should ensure that the keypoint is mapped accurately and corresponds to the same target in both stereo images;
    2. The annotator should also look back at the previous frame in the video sequence to ensure temporal video consistency in labelling;
    3. If it the keypoint is difficult to label, according to the definition above, then the annotator should set `is_difficult = True`. This can be done by pressing `m` to `m`ark the bounding box as a difficult one. You will notice a big red `\` drawn in the image when `is_difficult = True`.
4. Steps 1. to 3. should be repeated if you want to labell multiple keypoints per video. Once you are satisfied with the labeling of that keypoint in the entire stereo video, you can go back to the img 0, press `w` to select the next keypoint id, and go back to step 1. to start labeling the next keypoint.
5. The annotations should be reviewed by another annotator.
6. Finally, once the labelling is reviewed press `g` to generate the `g`round truth.

#### How to eliminate a bounding box?

First select the bounding box that you want to delete. By default the selected bounding box is shown in red. Then press `e` (standing for `e`liminate).

#### How to select a range of images?

If you want to set `is_visible_in_both_stereo = False` to a range of pictures you can again use `r` (standing for `r`ange). After pressing `r` just use `a` and `d` to select the desired range of images. Then press `v` to toggle the visibility. The same logic would apply vice-versa.

You can also use range to `e`liminate multiple images, `m`ark as `is_difficult`.

## Zoom mode

The middle mouse can be used for zoom-in and zoom-out of the images, however, it is more practical to use the zoom mode. The zoom mode allows you to labell faster by focusing on the area around the keypoints. Labell a pair of keypoints and you will notice a blue rectangle around them, if you press `z` (standing for `z`oom) you will zoom in or out of that blue rectangle. In zoom mode you can also re-adjust the bounding boxes by clicking again. Give it a try!

## How to run it?

I recommend you to create a Python virtual environment:

```
python3.9 -m pip install --user virtualenv
python3.9 -m virtualenv venv
```

Then you can activate that environment and install the requirements using:
```
source venv/bin/activate
pip install -r requirements.txt
```

Now, when the `venv` is activated you can run the code using:

```
python main.py
```
