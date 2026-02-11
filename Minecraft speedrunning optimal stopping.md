
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

##

