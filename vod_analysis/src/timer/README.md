## timer CNN

I made a tiny CNN to label the timer robustly instead of using OCR.

The timer has a fixed format
```makefile
xx:yy.zzz
```

So instead of runing general ocr on the whole string, we detect the timer ROI, split into fixed digit slots, run CNN on each digit, and reconstruct the formatted string.

## model.py

small CNN for 32x32 digits

The cool thing about this CNN is that it has 3 channels-- grayscale image, edge magnitude, and yellow hue masking. I thought the yellow and edge would be informative channels since the font is always the same with the outline and its yellow.

```python
Conv2d(3 -> 16) → ReLU -> MaxPool
Conv2d(16 -> 32) → ReLU -> MaxPool
Conv2d(32 -> 64) → ReLU
AdaptiveAvgPool -> Linear(64 → 10)

````

edge magnitude is only done on yellow masked regions. 


## what the utils do:

<h4>getting bounding box utils</h4>

**../../roi_calibrator.py** helps you draw the bounding boxes for timer and toast since in some of the vods its slightly different. creates roi.json in configs. 
- press t to define timer bounding box
- o to define the toast bounding box (achievements in the green text)
- s is to save
- use f/b to navigate through frames in the video to make sure its ok across the video.

**../scripts/calibrate_timer_digits.py** helps you mark the digit boundaries and get rid of the colon and period (they were super hard to classify and not worth it). creates timer_digit_bounds.json in configs.
- press n to get a new random frame.

**extract_random_timer_roi.py** randomly samples a frame from --video using roi.json defined in --roi. this is so you can check to see if your rois are ok and robust across timeframes. 




**build_timer_dataset.py** uses timer roi from configs/roi.json, samples random frames across the entire video, calls google vision on the roi, extracts per symbol boxes + confidence, saves preproccessed 32x32 glyph crops, and does this for 200 images per class 0-9
Samples without replacement until needed.
```python
python -m src.timer.build_timer_dataset_raw --video input_lowres.mp4 --roi configs/roi.json --timer_digits configs/timer_digit_bounds.json  --out_root data/timer_digits_raw --target 700 --val_ratio 0.1 --pad 0

```
- im generating a labeled dataset using expensive google vision api to make sure its nice and accurate without me having to label a bunch of shit.
- see the zip file in the data folder. open source ocr did very poorly and wasn't usable on the timer. - -10$ in api credits


**train_timer_cnn.py** is to train it. yellow_thr is the yellow threshold for the HSV masking

```python
python -m src.timer.train_timer_cnn --data_root data/timer_digits_raw --num_workers 0 --epochs 110 --yellow_thr 23
```

**../scripts/test_random_timer_digits.py**
takes random frame from video and outputs the bounding box/roi using configs

vision_pilot is just me testing out the google vision api

