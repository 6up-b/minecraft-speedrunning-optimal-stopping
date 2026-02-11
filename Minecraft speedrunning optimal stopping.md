
Speedruns can be defined as a multi stage stochastic process.

overworld --> nether --> fortress --> bastion trade --> stronghold --> end portal --> end fight

We learn the distribution of outcomes and conditional completion times from features we can derive from VODs.
From this we can discover novel optimal reset rules and have a Voyager-like autonomous agent complete speedruns
## Feature derivation

- Village time (maybe some kind of YOLO object detection)
- Ruined portal time (also object detection)
- Nether entry time (based on opencv template for purple pixels)
- bastion type (Bill Tin)
- first blaze count
- first pearl trades

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
1. t_world_load (first controllable frame, run start)
2. t_first_structure_seen (village/ruins/shipwreck)
3. t_nether_enter
4. t_bastion_seen
5. t_first_piglin_barter (count of barters)
6. t_fortress_seen
7. t_blaze_rod_obtained (count)
8. t_stronghold_enter
9. t_end_enter
10. t_dragon_death (run finished)

t_world_load


[[forsen vod screenshots for opencv]]
