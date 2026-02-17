# minecraft-speedrunning-optimal-stopping
speedruns as a multistate stochastic proccess


## todo:

<h6>getting a good dataset</h6>
- timer **((mostly)) DONE**
    - opencv + pytorch timer
    - segmentation + per char CNN 
    - small
    - important! trained on only 1 vod. bounding box i think differs across vods. So probably have to manually edit or make some high iq YOLO model to draw the bounding box on the timer instead of it being hardcoded in configs/roi.json + timer_digit_bounds.json
    - val_acc of around 99% for the CNN pretty tuff

- toast:
    - fixed ROI + OCR
- subtitle:
    - line detect + traclomg + OCR
    - CTC


note: think about how reset policy is fundamentally different from forced reset (death)
