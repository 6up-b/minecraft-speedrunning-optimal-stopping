# minecraft-speedrunning-optimal-stopping
speedruns as a multistate stochastic proccess

create your conda dev environment (the yml is in the vod_analysis folder)
```
conda env create -f environment.yml

```
surya installation
```
pip install -e extra_modules/surya

```

![[Pasted image 20260223163839.png]]
![[Pasted image 20260223163905.png]]
## todo:

#### getting a good dataset

- timer (go to vod_analysis/src/timer)

  - **timer pipeline:**
    1. check top right corner (top 30% xy)
    2. template match "IGT:"
    3. create bounding box for digits relative to position of IGT: (percentage)

  - **toast pipeline**
    1. check bottom 20% of screen (20% down, middle margin 15%)
    2. look for ui template match (inventory box)
    3. similarly use size of ui template match to find relative position of big toast text box
    4. green text color mask to find roi within big toast box then apply fine tuned OCR
       1. fine tune OCR on minecraft font
       2. synthetic dataset generated from minecraft font + add noise
       3. add some random screenshots too that are manually labeled

  - **inventory pipeline**
    1. start from ui template match
    2. check for epearl blaze rod template match

  - **every time digit confidence drops**
    - rerun full position pipelines for timer, toast and inventory

- toast:
  - fixed ROI + OCR
- make final test pipeline script that takes in vod and draws roi, inferences and plays back at 1-2 fps with j/l to jump frames, log output to json
- after pipeline is finished, need to manually check over some long vod (not seen before val) to see if roi are being plotted correctly, inferences are correct, everything is being logged in a useful way for analysis


toast template match should be rgb of hearts


note: think about how reset policy is fundamentally different from forced reset (death)

## autonomous agent
- bot client Mineflayer 
- macro skills (enter nether, loot bastion, go to fortress, barter, blind travel, locate stronghold, enter end, fight dragon)
- we have a macro skil library --> planner to choose which macro to execute next --> outcome/value estimates (probability of sub T, expected reamining time, failure risk/hazard), reset policy for model to decide to stop or continue
        - censored trajectory modeling? since we have very little dragon kills T_T
- use forsen's runs. ocr milestones to estimate distributions of stages time to nether, time to bbastion, etc, and condditional success rates for transitions
- partial trajectories P(reach next milestone), P(sub-T), expected remaining time
- ((weird idea)) learn route policy from forsen like how long to search before giving up, how often to reset given certain evidence. imagine an xqc route policy agent versus a forsen route policy agent


