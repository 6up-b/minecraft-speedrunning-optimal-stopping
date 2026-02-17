# minecraft-speedrunning-optimal-stopping
speedruns as a multistate stochastic proccess

create your conda dev environment (the yml is in the vod_analysis folder)
```
conda env create -f environment.yml

```


## todo:

<h4>getting a good dataset</h4>
- timer (go to vod_analysis/src/timer)
    - important! trained on only 1 vod. bounding box i think differs across vods. So probably have to manually edit or make some high iq YOLO model to draw the bounding box on the timer instead of it being hardcoded in configs/roi.json + timer_digit_bounds.json
		- **timer pipeline:**
			1. check top right corner (top 30% xy)
			2. template match "IGT:"
			3. create bounding box for digits relative to position of IGT: (percentage)
		- **toast pipeline**
			1. check bottom 20% of screen (20% down, middle margin 15%)
			2. look for ui template match (inventory box)
			3. similarly use size of ui template match to find relative position of toast text
			4. green text color mask then fine tuned OCR
				1. fine tune OCR on minecraft font
		- **inventory pipeline**
			1. start from ui template match
			2. check for epearl blaze rod template match
		- **every time digit confidence drops**
			- rerun full position pipelines for timer, toast and inventory
			- f
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


