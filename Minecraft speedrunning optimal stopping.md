


  Step 1: Extracted shared geometry (src/timer/geometry.py) — new file

  - Extracted clamp, clamp01, match_anchor_multiscale, timer_xyxy_from_anchor, crop_xyxy, digit_boxes_excluding_seps, letterbox, prep_digit_for_model, TimerLayout,
  load_timer_layout from infer.py
  - Updated infer.py to import from geometry.py + kept _-prefixed aliases for backward compat with extract_random_timer_roi.py and test_random_timer_digits.py

  Step 2: Manual labeling tool (scripts/label_timer_digits.py) — new file

  - Pre-extracts timer ROIs from random video frames using anchor-based layout
  - Shows upscaled timer ROI, user types 7 digits + Enter
  - Supports s=skip, q=quit, backspace, Esc to cancel
  - Saves letterboxed 32x32 BGR crops to data/timer_digits_v2/{train,val}/{0..9}/
  - Tracks class counts, auto-splits train/val

  Step 3: Google Vision dataset builder (src/timer/build_timer_dataset_from_anchor.py) — new file

  - Same geometry as labeling tool, but uses Google Vision API for automatic labeling
  - Supports --videos for multi-VOD extraction

  Step 4: Augmented dataset class (src/timer/datasets.py) — modified

  - Added TimerEdgeAugDataset with 6 augmentations on BGR before 3-channel preprocessing:
  scale jitter, translation, brightness/contrast, gaussian noise, rotation, JPEG artifacts
  - Added shared _bgr_to_3ch and _augment_bgr helpers

  Step 5: Model + training updates

  - model.py: Added Dropout2d(p) always present (p=0 is no-op), plus load_state_dict_compat() for old checkpoint remapping
  - train_timer_cnn.py: Added --augment (default True), --dropout (default 0.1), --save_misclassified, default --data_root data/timer_digits_v2, default --epochs 40,
  prints class distribution
  - pipeline.py and test_infer.py: Updated to use load_state_dict_compat()

  Workflow to retrain:

  cd vod_analysis

  # 1. Label digits from VODs
  python scripts/label_timer_digits.py --video input_lowres.mp4 --target 100
  python scripts/label_timer_digits.py --video part538.mp4 --target 100

  # 2. Train with augmentation
  python -m src.timer.train_timer_cnn --augment --epochs 40 --dropout 0.1

  # 3. Test
  python src/timer/test_infer.py --video input_lowres.mp4


****
**workflow to retrain**
  cd vod_analysis

  # 1. Label digits from VODs
  python scripts/label_timer_digits.py --video input_lowres.mp4 --target 100
  python scripts/label_timer_digits.py --video part538.mp4 --target 100

  # 2. Train with augmentation
  python -m src.timer.train_timer_cnn --augment --epochs 40 --dropout 0.1

  # 3. Test
  python src/timer/test_infer.py --video input_lowres.mp4
## todo
- fine tune ocr on minecraft text and numbers
- try out template matching to count number of 
- K

Speedruns can be defined as a multi stage stochastic process.

overworld --> nether --> fortress --> bastion trade --> stronghold --> end portal --> end fight

We learn the distribution of outcomes and conditional completion times from features we can derive from VODs.
From this we can discover novel optimal reset rules and have a Voyager-like autonomous agent complete speedruns
## Feature derivation
1. t_world_load (first controllable frame, run start)
	1. compute perceptual hash similarity between frames  (may be easier than detecting 00:00:000)
	2. or just use ocr 00:00:00xx with finetuned
2. t_nether_enter
	1. toast
3. t_bastion_seen
	1. toast
4. t_first_piglin_nearby
	1. subtitle
5. t_10_epearls
	1. template matching + hash similarity
6. t_fortress_seen
	1. toast
7. t_first_blaze_nearby
	1. subtitle
8. t_6_blazerods
	1. template matching + hash similarity
9. t_stronghold_enter
	1. toast
10. t_end_enter
	1. toast
11. t_dragon_death (run finished)
	1. toast
12. t_death
	1. template matching


crazy ideas:
- run type (village, shipwreck)
- bastion type detection (bill tin)
- 

## Optimal stopping for resets

When should you reset given partial information?

What policy minimizes expected time to personal best/wr? (maximize probability of pb per hour or minimize expected real time until first run under target time T)

Given a reset policy or a mixture distributions of reset policies, what is the expected time until you beat the record T?

How does policy change the rarity of success? (for example, xqc tends to reset before he enters the nether limiting him from Allah fortress+bastions in the nether. this is in contrast to forsen who usually continues the run despite a mid overworld. In other words, stricter resets(xqc) increase success probability conditional on continuing but reduce attempts per hour. Lenient resets(brute fors) increase attempt count but have lower per attempt success)

Can we create a Voyager like agent (autonomous) to implement our better reset + execution strategy better than humans?
- choose resets (meta policy)
- choose routes/actions within runs (micro policy)
- propose new optimizations ?? (trade routing, travel heuristics, risk controls??)


## Extracted structures

From forsen's vods
- observed prefix signals, features available by time t
- states: latent "quality" of seed/run (inferred, bayesian)
- time remaining distribution given observed prefix signals
- failure modes (deaths, no fortress, shit trades)

## Voyager summary + ideas for voyager-like agent

imagine livestreaming the agent trying to speedrun minecraft..........

Most important part is that Voyager controls the game via generating higher level primitives instead of controlling pixels or keystrokes. Then they use environment feedback, execution errors and self verification as feedback. 
what would be cool is visual perception since when this was published there was only text GPT4. I think text only summaries are lossy and structured vision detection (like bounding box fortress found or map features) would be really cool.  I think also another point of improvement is that Voyager uses embedding similarity + cosine nearest neighbor where instead we can use some cooler SOTA neural memory/retrieval policies. there was also a  lot of work done on world models to predict future states, rewards, events like trajectory transformers or video prediction models. We can probably use vision transformers to detect structures 

- Voayger has LLM propose next task based on world state, inventory, biome/time, nearby blocks/entitties, and past success/fail history. 
- Then given the new task Voyager forms a text query and retrieves top-k relevant prior skills from a vector database. 
- GPT4 writes a javascript function that calls provided low level minedojo primitives (explore, mineBlock, craftItem, killMob). 
- in the Minecraft sim its executed with emitted environment feedback (progress sumarry like need 8 iron) via chat +logs. captures execution errors also with invalid api calls and stack traces. 
- GPT4 self verifies by deciding if the task is completed and if not produces critique.
- repeat refinement for a few rounds. **if success, store the resulting code as a new skill in the library then query curriculum for next task**

The only embeddings they have are for skill retrieval and not for perception nor control.

Skill library: (executable program/code skill, embedding vector of the program description from GPT generated text)
at query time they embed a query text (task plan + environment feedback context) and retrieve top-k most similar skills to include in next prompt

For text embeddings they used some OpenAI text-embedding-ada-002

So text-->embedding vector via embedding API --> nearest neighbor search in vector space to retrieve top-k skills --> concatenation with retrieved code snippets into the prompt so GPT4 can compose /reuse them

**Voyager data**
Not really trained on a static labeled dataset. Data are the following:
- trajectories of interaction (sequence sof state summaries, tasks, generated code, execution log environment feedback, errors, etc)
- Skill artifacts from code that ends up succeeding + natural language description
Self verification from GPT4 produces the success/fail label for each attempted task with critique text. Environment feedback is produced during execution and is kind of used as a structured ish supervision for debugging. 

[twitch chat downloader + twitch vod downloader](https://github.com/lay295/TwitchDownloader?tab=readme-ov-file)

[minecraft speedrunning guide for 1.16.1 forsen](https://github.com/Metacor/Minecraft-Speedrun-Guide?tab=readme-ov-file)

[forsen minecraft vods](https://www.youtube.com/watch?v=xNTWpdiyEpo&list=PLqK_XCyJ557rlSpCKAD7UmvgyR2IkCs1J)


[slowrunning 23:38](https://www.youtube.com/watch?v=9VvhmZX_JgQ&list=PLqK_XCyJ557rlSpCKAD7UmvgyR2IkCs1J&index=200)
## markers

1. spawn in village
2. search for lava lake or ruined portal
3. enter nether
4. find bastion
5. trade gold to piglins for obsidian and epearls
6. find fortress
7. kill blaze for blaze rods
8. exit nether
9. enter stronghold (this is where they use the cool Ninjabrain triangulation method to find it)
10. find portal room + enter the end
11. kill the ender dragon

## timestamps


t_world_load


[[forsen vod screenshots for opencv]]
