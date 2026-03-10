
## 1. Anchor-relative layout calibration

### `src/timer/calibrate_timer_layout_from_anchor.py`

This is the one time calibration step.

It does three things:

1. Loads a frame from a VOD
2. Finds the **`IGT:  ` anchor** using the anchor template PNG
3. Lets you define the timer ROI and digit boundaries **relative to the anchor**

What gets saved:

* `configs/timer_layout_from_anchor.json`

What that JSON contains:

* anchor template metadata
* anchor search region (`top_frac`, `right_frac`)
* `timer_roi_rel_to_anchor`
* `digit_bounds_x_norm_within_timer_roi`
* `separators_x_norm_within_timer_roi`

After this step, you find the anchor via templating and digit positions are derived from from the anchor ("IGT: ").

---

## 2. Visual ROI sanity check

### `src/timer/test_anchor_timer_rois.py`

This is the geometry/debug script.

It:

1. Loads `configs/timer_layout_from_anchor.json`
2. Loads the anchor PNG
3. Randomly samples a frame from a VOD
4. Finds the anchor by template matching
5. Reconstructs:

   * anchor bbox
   * timer bbox
   * 7 digit boxes
6. Draws them on the frame
7. Saves cropped digit images

Outputs:

* `overlay.png`
* `timer_roi.png`
* `digit_0.png` through `digit_6.png`
* `meta.json`

This is used to confirm that the anchor match and relative digit geometry still make sense across different VODs.

---

## 3. Build the labeled digit dataset

### `src/timer/build_timer_dataset_digits_from_vision.py`

This is the dataset generation step.

It:

1. Opens a VOD
2. Samples random frames
3. Crops the timer ROI
4. Uses **Google Vision API** on the full timer ROI to read the timer text
5. Uses the anchor-relative digit layout to crop the 7 digits
6. Uses the OCR output string to label those digit crops
7. Saves the digits into digit-class folders

Note:

* It only saves digits `0..9`
* The separator bands are used to exclude colon/dot from neighboring digit crops

Typical output:

* `data/timer_digits_raw/train/0..9/...`
* `data/timer_digits_raw/val/0..9/...`

So this file turns the VOD into a digits only supervised training set.

---

## 4. Prepare model inputs

### `src/timer/datasets.py`

This defines `TimerEdgeDataset`.

Its job is to take saved digit crops and convert them into the model input tensor.

Current version produces **3 channels**:

1. raw grayscale
2. edge magnitude from yellow-masked grayscale
3. yellow mask itself

Each digit image becomes a tensor shaped:

* `[3, 32, 32]`


---

## 5. Define the CNN

### `src/timer/model.py`

This defines `TinyCharCNN`.

* digit-only classifier
* classes are `0..9`
* input channels = 3

So the model predicts one of ten digits for each cropped slot.

---

## 6. Train the digit classifier

### `src/timer/train_timer_cnn.py`

This trains the timer digit model.

It:

1. Loads the dataset from `data/timer_digits_raw/train` and `val`
2. Uses `TimerEdgeDataset` to build 3-channel inputs
3. Instantiates `TinyCharCNN(in_channels=3, num_classes=10)`
4. Trains the model
5. Prints top confusions
6. Saves the best checkpoint

Output:

* `timer_model.pth`

This is the final learned digit recognizer used at inference time.

---

## 7. Optional decoding utilities

### `src/timer/decode.py`

This file implements stateless constrained decoding of the 7 timer digits predicted by the CNN. 
The model outputs logits of shape:

`[7, 10]`

corresponding to the digits:

`d0 d1 : d2 d3 . d4 d5 d6`

which represent the timer:

`mm:ss.mmm`

This file converts the raw logits into a valid timer string while enforcing known timer structure.

The constraints currently implemented are:

`d2 ∈ {0..5}` (seconds tens place)

optionally `d0 ∈ {0..5}` (minutes tens place)

These constraints are applied directly to the logits by setting invalid digits to -inf.
The first pass performs a fast constrained greedy decode (softmax probabilities, highest probability digit) then uses ambiguity aware top-k decoding. 

The goal is to improve full timer reconstruction even when some individual digit logits are ambiguous.

---

## 8. Runtime inference

### `src/timer/infer.py`

This is the main runtime inference file.

Given a frame, it does:

1. Load `configs/timer_layout_from_anchor.json`
2. Load the anchor template PNG
3. Find the anchor in the frame via template matching
4. Use `timer_roi_rel_to_anchor` to crop the timer ROI
5. Use `digit_bounds_x_norm_within_timer_roi` to crop 7 digit slots
6. Use separator bands to exclude colon/dot from adjacent digits
7. Convert each digit crop into the 3-channel model input
8. Run `timer_model.pth`
9. Reconstruct `xx:yy.zzz`
10. Return:

* text
* seconds
* overall confidence
* anchor bbox
* timer bbox
* digit bboxes
* per-digit confidences

This is the file the full VOD-reading pipeline will call per frame.

And this is where the future “re-find anchor if confidence drops” logic naturally belongs:

* if digit confidence falls below threshold, rerun anchor matching
* but still use the same relative layout JSON after the anchor is found

---

## 9. End-to-end inference test

### `src/timer/test_infer.py`

This is an integration tester.

It does the following:

1. Load a video
2. Randomly sample one or more frames
3. Load:

   * `configs/timer_layout_from_anchor.json`
   * anchor template PNG
   * `timer_model.pth`
4. Call `infer_timer(...)` from `infer.py`
5. Print:

   * predicted timer string
   * seconds
   * overall confidence
   * digit confidences
6. Save an annotated debug image showing:

   * anchor bbox
   * timer bbox
   * digit boxes
   * inferred timer string

---

# Summary file flow

## Calibration / geometry

* `calibrate_timer_layout_from_anchor.py`
* `test_anchor_timer_rois.py`

## Dataset creation

* `build_timer_dataset_digits_from_vision.py`

## Training

* `datasets.py`
* `model.py`
* `train_timer_cnn.py`


## Inference

* (optionally) `decode.py`
* `infer.py`
* `test_infer.py`

---

# Core dependency chain

## At calibration time

* `anchor_png` + VOD frame
  → `configs/timer_layout_from_anchor.json`

## At training time

* `configs/timer_layout_from_anchor.json` + Google Vision
  → `data/timer_digits_raw`
  → `train_timer_cnn.py`
  → `timer_model.pth`

## At inference time

* `configs/timer_layout_from_anchor.json`
* `anchor_png`
* `timer_model.pth`
* input frame
  → `infer.py`
  → timer text + confidence


## notes on timer CNN

I made a tiny CNN to label the timer robustly instead of using OCR.

The timer has a fixed format
```makefile
xx:yy.zzz
```

So instead of runing general ocr on the whole string, we detect the timer ROI, split into fixed digit slots, run CNN on each digit, and reconstruct the formatted string.

idea:
instead of greedy argmax per digit, use top k logits per digit and choose best valid time under constraints
For example, for each digit slot i, get top 3 candidates with log prob. then enumerate combinations but only keep valid ones such that digit2 is less than digit 5.
pick max total logprob
This inbuilt sanity check will probably make the errors super low


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

