# minecraft-speedrunning-optimal-stopping
speedruns as a multistate stochastic proccess


## todo:

<h4>getting a good dataset</h4>
- timer (go to vod_analysis/src/timer)
    - opencv + pytorch timer
    - segmentation + per char CNN 
    - small
    - important! trained on only 1 vod. bounding box i think differs across vods. So probably have to manually edit or make some high iq YOLO model to draw the bounding box on the timer instead of it being hardcoded in configs/roi.json + timer_digit_bounds.json
    - val_acc of around 99% for the CNN. surely not overfitting

- toast:
    - fixed ROI + OCR
- subtitle:
    - line detect + traclomg + OCR
    - CTC


note: think about how reset policy is fundamentally different from forced reset (death)

## autonomous agent
- bot client Mineflayer 
- macro skills (enter nether, loot bastion, go to fortress, barter, blind travel, locate stronghold, enter end, fight dragon)
- we have a macro skil library --> planner to choose which macro to execute next --> outcome/value estimates (probability of sub T, expected reamining time, failure risk/hazard), reset policy for model to decide to stop or continue
        - censored trajectory modeling? since we have very little dragon kills T_T
- use forsen's runs. ocr milestones to estimate distributions of stages time to nether, time to bbastion, etc, and condditional success rates for transitions
- partial trajectories P(reach next milestone), P(sub-T), expected remaining time
- ((weird idea)) learn route policy from forsen like how long to search before giving up, how often to reset given certain evidence. imagine an xqc route policy agent versus a forsen route policy agent


