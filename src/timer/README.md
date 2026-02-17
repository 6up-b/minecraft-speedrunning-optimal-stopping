## timer CNN

I made a tiny CNN to label the timer robustly instead of using OCR.

## what the utils do:

extract_random_timer_roi.py randomly samples a frame from --video using roi.json defined in --roi. this is so you can check to see if your rois are ok and robust across timeframes.

im generating a labeled dataset using expensive google vision api to make sure its nice and accurate without me having to label a bunch of shit.

vision_pilot is just me testing out the google vision api

roi_calibrator.py helps you draw the bounding boxes for timer and toast since in some of the vods its slightly different. Majority of 2025 vods use the current roi.json so you probably won't have to change anything wrt to that.

build_timer_dataset.py uses timer roi from configs/roi.json, samples random frames across the entire video, calls google vision on the roi, extracts per symbol boxes + confidence, saves preproccessed 32x32 glyph crops, and does this for 200 images per class 0-9, :, .
Samples without replacement until needed.

