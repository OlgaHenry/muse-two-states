# Two mental states with a Muse

A small brain-computer interface by Olga Henry: two mental states turn a cross red or blue,
using only the four EEG sensors of a Muse headband. Nothing on the screen moves.

- **RED:** counting backwards (for example 734, 727, 720…), effortful mental work
- **BLUE:** recalling a pleasant memory in detail

On my own data the pair separated at 86% offline on single 8 s trials (38 trials, the features used here); in the live game I got
21 of 24 on one day and 9 of 9 a few days later with no new calibration (the color video).

## Start here

1. Install (see **Install** below) and find your headband: `python find_muse.py`
2. **First run: try my model before anything else**, while you have no trials of your own yet:

   ```
   python two_states.py --user yourname --board s --start-from olga_model.json
   ```

   Sensor check, instructions, four warm-up trials, two of each (no feedback, they measure your signal levels),
   then **block 1 with my model and no calibration**: that is the transfer test. Those 16 trials
   also count as your calibration, so **blocks 2 and 3 already use a model trained on you**.
   The result of every block is written to `summary_yourname.txt` (which model, how many hits),
   every trial to `trials_yourname_log.csv`.
3. **Any later day:** `python two_states.py --user yourname --board s` goes straight to the game,
   trained on all your trials so far. Add `--calibrate` for a fresh calibration round on a bad day.
4. **Free mode**, once you have a model: `python two_states.py --user yourname --board s --free`

Use `--board athena` for a Muse S Athena and `--board 2` for a Muse 2. Q quits at any moment and
everything is saved.

## How it works

1. **Sensor check:** a black screen shows each of the four sensors live (good / no contact /
   floating). This is a rough signal check (samples stuck at the extremes, alternating patterns,
   much noisier than the others), not an impedance measurement; it was tuned on the Athena. Adjust the headband, press SPACE when it looks good.
   The same check comes back in every break.
2. **Calibration** (first time, or with `--calibrate`): 16 trials of 8 s, half of each task, in a
   random order, about 4 to 5 minutes. The cross does not respond yet.
3. **Training:** a shrinkage LDA on 3 s windows. Features per channel, from 4 Hz up: theta and alpha
   log power and the 1/f background (slope and level, alpha band left out of the fit). Beta is left out
   by default because on the forehead it is mostly muscle (`--beta` adds it). The forehead eye signals
   (AF7 − AF8 and AF7 + AF8) are regressed out of the two ear channels.
4. **Game:** 3 blocks of 12 trials. After every block the model is retrained on all your trials, so
   you adapt to the feedback and the model adapts to you.
5. **Your trials are kept** in `trials_<user>.npz` and written after every single trial. Your model is
   trained only on your own trials; my model is used only in the `--start-from` block.

In my recordings the difference between the two states turned out to be mostly a tilt of the 1/f
background behind the right ear (more slow, less fast activity while remembering), not an alpha peak.

## How the model learns from your trials (and what to do with bad ones)

- Every trial is stored with the task that was **on the screen** (count = red, memory = blue), not
  with whether it was a hit. Hits and misses are never used for training.
- So the model **believes every label**: if the screen said "count" and your mind wandered, it still
  learns that trial as "count". A few such trials only blur the boundary; many of them teach it noise.
- **Press X right after a trial** (in the 3 to 5 s pause before the next cue, or in the break that
  follows) if you were not really in the state; once the next trial starts, X no longer works. That trial stays in the file but is never used for training. Only the last
  trial can be marked.
- **Test runs:** if you are just trying the program out and not concentrating, use a separate user
  name (for example `--user test`), so those trials never mix with your real ones.
- **Day to day:** the signal shifts between days more than between the two states. So every run of
  the program is first centered on its own average before training (a run needs at least 2 good
  trials of each kind for that). At the start of a new day, until the game has 2 trials of each kind,
  the previous run's average is used; from then on the current run's.
- After every block of 12 the model is retrained on all your good trials from all days.

## Install

You need Python 3.10 or newer with tkinter (the python.org installer has it; with Homebrew Python
run `brew install python-tk`), and a terminal app that is allowed to use Bluetooth (macOS: System
Settings > Privacy & Security > Bluetooth, allow Terminal or iTerm). Then:

```
python3 -m venv ~/muse-venv
source ~/muse-venv/bin/activate          # every new terminal: activate it again (Windows: muse-venv\Scripts\activate)
pip install -r requirements.txt
```

The commands below assume the environment is active, so plain `python` is the right one. BrainFlow is
pinned to 5.22.2, the version this was tested with (older versions may not know the Athena). On Linux,
tkinter is a separate package (`sudo apt install python3-tk`).

Voice prompts use the macOS `say` command; on other systems the prompts are only printed and shown.
I have only run it on a Mac with a Muse S Athena.

**With Claude Code** you can paste this: "Set up this repository: make a virtual environment, install
requirements.txt, check that tkinter imports, run find_muse.py, then run `two_states.py --user test --board s --dry
--mute --windowed` to check that everything starts."

## Find your headband

Turn the Muse on (it blinks when it is free to connect; a steady light means it is already
connected to the phone app or another computer, so close the app), then:

```
~/muse-venv/bin/python find_muse.py
```

It lists names like `MuseS-0D57` with their addresses (a long UUID on a Mac, a MAC address on
Windows and Linux). Use `--name MuseS-0D57` or `--address <address>`; without either, BrainFlow
connects to the first Muse it finds.

## Run

```
python two_states.py --user yourname --board s --name MuseS-XXXX   # first time: calibration, then the game
python two_states.py --user yourname --board s --name MuseS-XXXX   # any later day: straight to the game, on all your earlier trials
python two_states.py --user yourname --board s --calibrate         # add a fresh calibration round first
python two_states.py --user yourname --board s --free                           # free mode, see below
python two_states.py --user test --board s --dry --mute --windowed              # no headband, synthetic data
```

`--board` is required: `athena` for the Muse S Athena, `s` for the Muse S (gen 1 / 2), `2` for the Muse 2; `--mains 50` for 50 Hz countries, `--windowed`
for a window instead of full screen. It talks to the Muse directly over Bluetooth through BrainFlow;
there is no LSL in it.

**Quitting:** press **Q** at any moment. Everything recorded so far is saved: your trials
(`trials_<user>.npz`, written after every trial), your model (`model_<user>.pkl`) and the raw
recording (`raw_<user>_<time>.csv`). SPACE continues after a break.

## Free mode: get into the state first, then press SPACE

This is how I made the video. In the calibration and the game the program tells you which state to
do and starts the 8 seconds itself. In free mode **you** choose: I drew a card (red or blue), started
counting or remembering, waited until I was really in it (it is easy to get distracted at the
start, but once I had started, I had started), and only then pressed SPACE. Because you are already in the state, the program reads the last
3 s straight away, with no wait. From that moment the
cross turns red or blue step by step, every half second, toward the state it reads; when a color
is reached the whole screen fills with it, or it gives up after 20 s. Free attempts are not used for training. Free mode needs a trained
model, so play the calibration and the game first.

## Transfer test: start from my model

```
python two_states.py --user yourname --board s --start-from olga_model.json
```

`olga_model.json` is my trained model: 12 numbers (the direction that separated my two states, over
AF7, AF8 and TP10) and nothing else, no recordings. Four warm-up trials without feedback (two of each) measure your own
signal levels; then my direction is applied to your features scaled to you. That tests whether my
model works on you with no calibration. Your trials are stored as usual, and once you have 12 the game
switches to a model trained on you. My model was trained on a Muse S Athena. Your features are scaled to
you, which takes care of different signal units between headband models, but the sensors and
their contact still differ, so treat a miss on another model as a question, not an answer.

Only the Muse S Athena has been tested. The Muse S and Muse 2 settings use BrainFlow's board ids and
should work, but the contact check was tuned on the Athena.

## Honest notes on the numbers

- The "held-out accuracy" shown before a block is a **rough** estimate: whole trials are held out,
  but the eye regression and each run's average were computed on all trials, and it scores the
  average of each 8 s trial, while the game accumulates small steps every half second. So it is not
  the expected hit rate; the hits in `summary_<user>.txt` are the real thing.
- My 86% comes from tasks I picked on my own data, so for anyone else it is a starting point, not a
  promise.
- Feedback starts once a full 3 s window of the trial is available (the model was trained on 3 s
  windows), so the first 4 s of each trial show no color.
- The eye regression is used only when both forehead sensors are in contact.

## Files it writes

`trials_<user>.npz` (your trials, written after every trial; dry runs go to `trials_<user>_dry.npz`),
`trials_<user>_log.csv` (every trial and every X), `summary_<user>.txt` (hits per block),
`model_<user>.pkl`, `raw_<user>_<time>.csv` (the raw recording with markers: 1 red trial start,
2 blue trial start, 9 trial end, 7 marked with X). A user name is tied to one headband model; use a
new name for another model.

## Tips

- Eyes on the cross, jaw and face relaxed, lips still. The cross never moves, so there is nothing
  for the eyes to follow.
- Good contact behind the ears matters most. If a sensor shows ✗ on the contact screen, adjust the
  headband, or slightly wet the sensors (or the skin and hair under them) with a little water.
- It works much worse when tired: late in the evening my two states dropped to chance.
- The signal shifts from day to day; the program centers every run on its own average (see above).

## If it does not work for you

These two states are the most different pair **for me**. Other people may need a different pair.
What I tried, so you can pick candidates (all with eyes open on a cross):

| state | how it came out for me |
|---|---|
| counting backwards by 7 | the "work" end, best partner for memory |
| recalling a pleasant memory | the "inward" end |
| resting, just looking at the cross | close to memory; rest vs counting was 84% offline |
| walking through my home in my mind | close to memory |
| imagining a melody | between, closer to memory |
| silently generating words | separated from both, but with the most facial muscle activity, probably silent speech |
| listening to a steady tone | close to counting |
| tone plus squeezing a fist | close to counting (adds muscle) |
| attention to breathing at the nose | looked different from both in only 3 trials, not confirmed |
| imagining falling asleep (short trials) | inconsistent: once different from both, once not |
| attending to my heartbeat | too few trials to say |
| an imagined beat or accent on a metronome | not visible in the EEG; it leaked into tiny head movements |

Almost everything fell on one axis from "effortful work" to "relaxed imagery", which is why a third
state has been hard. Happy to help you find your own pair.
