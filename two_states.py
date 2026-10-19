"""Two mental states -> red / blue with a Muse headband.

RED = counting backwards (effortful work), BLUE = recalling a pleasant memory.
Black screen with a cross that never moves: only its color changes (white -> red or blue). A live sensor-contact check at the start and in every break.

Three ways to run it (your trials are kept per user in trials_<user>.npz):
  first time        calibration (16 trials of 8 s, the cross does not respond), then the game
  any later day     the game straight away, trained on all your earlier trials
  --calibrate       add a fresh round of 16 calibration trials first (e.g. a new day, a bad day)
  --start-from olga_model.json
                    play with Olga's model, no calibration: the transfer test. Two warm-up
                    trials (no feedback) measure your own signal levels; her model's direction
                    is then applied to your features scaled to you. Your trials are stored, and
                    once you have 12 of them the game switches to your own model.
After every block of 12 the model is retrained on all your good trials; each run is first centered
on its own average (the signal shifts from day to day). X right after a trial = I was not in the
state, never train on it. Q quits at any moment; everything recorded so far is saved (the bank is written after every trial).

What the model looks at (per 3 s window, from 4 Hz up): theta and alpha log power and the
1/f background (slope and level) of each channel; beta is left out by default because on the
forehead it is mostly muscle (--beta adds it). The forehead eye signals (AF7 - AF8, AF7 + AF8)
are regressed out of the ear channels, and nothing on screen moves, so gaze has nothing to follow.

    python two_states.py --user cedric --board athena
    python two_states.py --user cedric --board 2
    python two_states.py --user cedric --board athena --start-from olga_model.json
    python two_states.py --user cedric --board athena --free           # free mode, like the color video
    python two_states.py --user test --board s --dry --mute --windowed # no headband

Free mode (how Olga made the video): YOU pick the state (e.g. draw a card), get into it first,
and only then press SPACE. The cross then turns red or blue step by step; when one color is
reached the whole screen fills with it (as in Olga's video), or it gives up after 20 s. Free attempts are not used for training. Needs a trained model (your own,
or --start-from).
--name MuseS-XXXX finds that headband by its name (see find_muse.py); --address uses an address directly.
"""

import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import time
import tkinter as tk
import traceback

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

ap = argparse.ArgumentParser()
ap.add_argument("--user", default="me")
ap.add_argument("--board", choices=["athena", "s", "2"], required=True,
                help="athena = Muse S Athena, s = Muse S (gen 1 / 2), 2 = Muse 2")
ap.add_argument("--address", default="")
ap.add_argument("--name", default="")
ap.add_argument("--free", action="store_true")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--start-from", default="")
ap.add_argument("--mains", type=int, default=60)
ap.add_argument("--beta", action="store_true")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--mute", action="store_true")
ap.add_argument("--windowed", action="store_true")
A = ap.parse_args()

BOARD = {"athena": BoardIds.MUSE_S_ATHENA_BOARD, "s": BoardIds.MUSE_S_BOARD, "2": BoardIds.MUSE_2_BOARD}[A.board]
if A.dry:
    BOARD = BoardIds.SYNTHETIC_BOARD
FS = BoardShim.get_sampling_rate(BOARD)
EEG = BoardShim.get_eeg_channels(BOARD)[:4]        # Muse order: TP9, AF7, AF8, TP10
NAMES = ["TP9  left ear", "AF7  forehead left", "AF8  forehead right", "TP10 right ear"]

CAL_TRIALS = 4 if A.dry else 16
GAME_TRIALS = 4 if A.dry else 12
GAME_BLOCKS = 1 if A.dry else 3
TRIAL, SKIP, WINDOW, STEP = 8.0, 1.0, 3.0, 0.5
GAIN = 0.16
PAUSE = (3, 5)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BANK = "trials_%s.npz" % A.user
SEG_N = int(TRIAL * FS)
MEMORIES = ["Recall a warm day at the beach", "Recall a dinner with people you love",
            "Recall a moment when everything worked out", "Recall your favorite walk",
            "Recall a gift that made you really happy"]


def task_text(side):
    if side == "left":
        return "Count backwards from %d by %d" % (random.randint(150, 990), random.choice([3, 4, 6, 7, 8, 9]))
    return random.choice(MEMORIES)


VOICE = ""
if shutil.which("say"):
    # an English voice on purpose: the Mac's default voice may be another language
    voices = os.popen("say -v '?'").read()
    VOICE = next((v for v in ("Samantha", "Alex", "Daniel", "Karen", "Moira") if "\n" + v + " " in "\n" + voices), "")


def say(text):
    print(">>", text)
    if not A.mute and shutil.which("say"):
        os.system('say %s "%s"' % (("-v " + VOICE) if VOICE else "", text.replace('"', '')))


def shuffle_limited(items, max_run=2):
    """Random order, at most two of the same in a row (strict alternation would be guessable)."""
    items = list(items)
    for _ in range(20000):
        random.shuffle(items)
        run, ok = 1, True
        for a, b in zip(items, items[1:]):
            run = run + 1 if a == b else 1
            if run > max_run:
                ok = False
                break
        if ok:
            return items
    return items

# ---------------------------------------------------------------- signal


sos = butter(4, [1, 45], btype="band", fs=FS, output="sos")
eye_sos = butter(4, [0.3, 5], btype="band", fs=FS, output="sos")
b_notch, a_notch = iirnotch(A.mains, 30, FS)


def prep(x):
    x = x - x.mean(axis=1, keepdims=True)
    return filtfilt(b_notch, a_notch, sosfiltfilt(sos, x, axis=1), axis=1)


def eye_signals(x):
    return np.stack([x[1] - x[2], (x[1] + x[2]) / 2])


def fit_eye_coefs(segments):
    big = np.concatenate([sosfiltfilt(eye_sos, s - s.mean(axis=1, keepdims=True), axis=1) for s in segments], axis=1)
    e = eye_signals(big)
    return {ch: np.linalg.lstsq(e.T, big[ch], rcond=None)[0] for ch in (0, 3)}


def remove_eyes(x, coefs):
    x = x.copy(); e = eye_signals(x)
    for ch in (0, 3):
        x[ch] -= coefs[ch] @ e
    return x


def features(x, channels):
    f, p = welch(x[channels], fs=FS, nperseg=min(x.shape[1], 2 * FS))
    bands = [(4, 8), (8, 13)] + ([(13, 30)] if A.beta else [])
    out = [np.log(p[:, (f >= lo) & (f < hi)].mean(axis=1)) for lo, hi in bands]
    m = (f >= 4) & (f <= 35) & ~((f >= 7) & (f <= 14))           # 1/f fit, alpha left out
    slope, level = np.polyfit(np.log10(f[m]), np.log10(p[:, m]).T, 1)
    return np.concatenate(out + [slope, level])


def windows_of(seg, coefs, channels):
    clean = remove_eyes(prep(seg), coefs)
    n, step = int(WINDOW * FS), int(STEP * FS)
    return [features(clean[:, a:a + n], channels) for a in range(int(SKIP * FS), clean.shape[1] - n + 1, step)]


def stuck(x):
    """Share of samples sitting at the window's extreme values: a channel off the skin saturates
    (sits at the rails) or alternates between them."""
    if len(x) == 0:
        return 1.0
    lo, hi = x.min(), x.max()
    return float(np.mean((x == lo) | (x == hi)))


def train(segs, labels, sessions, channels):
    """Shrinkage LDA on all good trials. Each session (one run of the program) is first centered on
    its own average, because the signal shifts from day to day more than between the two states
    (Olga's finding); a session needs at least 2 trials of each kind to have its own average,
    otherwise the average of all trials is used."""
    rails = [np.mean([stuck(x[ch]) for x in segs]) for ch in range(4)]
    channels = [ch for ch in channels if rails[ch] < 0.05]
    if not channels:
        raise SystemExit("no channel was in contact in the stored trials")
    coefs = fit_eye_coefs(list(segs))
    W = [np.array(windows_of(x, coefs, channels)) for x in segs]
    overall = np.concatenate(W).mean(0)
    means = {}
    for ses in set(sessions):
        idx = [i for i, t in enumerate(sessions) if t == ses]
        labs = [labels[i] for i in idx]
        if labs.count(0) >= 2 and labs.count(1) >= 2:
            means[ses] = np.concatenate([W[i] for i in idx]).mean(0)
    centre = lambda i: means.get(sessions[i], overall)
    X = np.concatenate([W[i] - centre(i) for i in range(len(W))])
    y = np.concatenate([[labels[i]] * len(W[i]) for i in range(len(W))])
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    model.fit(X, y)
    scale = float(np.std(model.decision_function(X))) or 1.0
    TX = np.array([(W[i] - centre(i)).mean(0) for i in range(len(W))])
    k = max(2, min(5, int(min(np.sum(labels == 0), np.sum(labels == 1)))))
    acc = cross_val_score(make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                          TX, labels, cv=StratifiedKFold(k, shuffle=True, random_state=0)).mean()
    latest = max(means) if means else None
    return {"model": model, "coefs": coefs, "channels": channels, "scale": scale, "cv_acc": acc, "n": len(labels),
            "beta": A.beta, "means": means, "overall": overall,
            "offset": means.get(STAMP, means[latest] if latest else overall)}


def session_offset(live, bank):
    """This run's own average as soon as it has 2 good trials of each kind; until then the
    latest earlier session's average."""
    idx = [i for i in range(len(bank["labels"])) if bank["sessions"][i] == STAMP and not bank["bad"][i]]
    labs = [bank["labels"][i] for i in idx]
    if labs.count(0) >= 2 and labs.count(1) >= 2:
        return np.concatenate([windows_of(bank["segs"][i], live["coefs"], live["channels"]) for i in idx]).mean(0)
    return live["offset"]


class Transferred:
    """Someone else's model applied to you: their LDA direction, YOUR feature scaling
    (running mean and spread of every window you produced in this session)."""

    def __init__(self, path):
        m = json.load(open(path))                    # plain numbers, nothing executable
        self.w, self.b = np.array(m["lda_coef"]), float(m["lda_intercept"])
        self.channels, self.scale, self.cv_acc, self.n = m["channels"], m["scale"], m["cv_acc"], m["n"]
        if m["beta"] != A.beta:
            raise SystemExit("this model was trained %s beta; run with%s --beta" % ("with" if m["beta"] else "without", "" if m["beta"] else "out"))
        self.seen = []
        self.coefs = None

    def add(self, feats):
        self.seen.extend(feats)

    def decision_function(self, X):
        S = np.array(self.seen); mu, sd = S.mean(0), S.std(0) + 1e-9
        return ((np.asarray(X) - mu) / sd) @ self.w + self.b


def load_bank():
    if os.path.exists(BANK):
        z = np.load(BANK, allow_pickle=True)
        n = len(z["labels"])
        return {"segs": list(z["segs"]), "labels": list(z["labels"]),
                "sessions": list(z["sessions"]) if "sessions" in z else ["old"] * n,
                "bad": list(z["bad"]) if "bad" in z else [False] * n}
    return {"segs": [], "labels": [], "sessions": [], "bad": []}


def save_bank(bank):
    np.savez(BANK, segs=np.array(bank["segs"]), labels=np.array(bank["labels"]),
             sessions=np.array(bank["sessions"]), bad=np.array(bank["bad"]))


def good(bank):
    """Trials that count for training: everything not marked with X."""
    idx = [i for i in range(len(bank["labels"])) if not bank["bad"][i]]
    return (np.array([bank["segs"][i] for i in idx]), np.array([bank["labels"][i] for i in idx]),
            [bank["sessions"][i] for i in idx])


# ---------------------------------------------------------------- screen


class Screen:
    def __init__(self):
        self.root = tk.Tk(); self.root.configure(bg="black")
        if A.windowed:
            self.root.geometry("1100x700")
        else:
            self.root.attributes("-fullscreen", True)
        self.root.bind("<KeyPress-q>", lambda e: setattr(self, "stopped", True))
        self.stopped = False
        self.front()
        self.root.update()
        self.W, self.H = self.root.winfo_width(), self.root.winfo_height()
        self.c = tk.Canvas(self.root, width=self.W, height=self.H, bg="black", highlightthickness=0); self.c.pack()
        self.cx, self.cy = self.W / 2, self.H / 2
        self.reach = self.W * 0.35
        self.cross = self.c.create_text(self.cx, self.cy, text="+", fill="white", font=("Helvetica", 96))
        self.top = self.c.create_text(self.cx, self.H * 0.18, text="", fill="#cccccc", font=("Helvetica", 32), justify="center")
        self.bottom = self.c.create_text(self.cx, self.H * 0.84, text="", fill="#888", font=("Helvetica", 22), justify="center")
        self.quit_hint = self.c.create_text(self.W - 20, self.H - 20, text="Q: quit (everything is saved)", fill="#555",
                                            font=("Helvetica", 14), anchor="se")
        self.pump()

    def front(self):
        """Bring the window in front of the Terminal and give it the keyboard (her remark: it opened
        behind the Terminal, so SPACE and Q went to the Terminal instead)."""
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(1500, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def fill(self, color):
        self.root.configure(bg=color); self.c.configure(bg=color); self.pump()

    def pump(self):
        self.root.update()
        if self.stopped:
            raise KeyboardInterrupt

    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.pump(); time.sleep(0.01)

    def wait_for_space(self):
        self.root.focus_force()
        self.space = False
        self.root.bind("<space>", lambda e: setattr(self, "space", True))
        while not self.space:
            self.pump(); time.sleep(0.02)
        self.root.unbind("<space>")

    def text(self, top=None, bottom=None):
        if top is not None: self.c.itemconfig(self.top, text=top)
        if bottom is not None: self.c.itemconfig(self.bottom, text=bottom)
        self.pump()


    def show(self, pos):
        """pos -1 (red) .. +1 (blue); 0 = white. The cross never moves."""
        a = abs(pos); r = 255 if pos <= 0 else int(255 * (1 - a)); b = 255 if pos >= 0 else int(255 * (1 - a))
        self.c.itemconfig(self.cross, fill="#%02x%02x%02x" % (r, int(255 * (1 - a)), b))
        self.pump()

    def contact(self, board):
        """Live sensor check (black screen): SPACE when it looks good."""
        self.root.focus_force()
        self.space = False
        self.root.bind("<space>", lambda e: setattr(self, "space", True))
        self.c.itemconfig(self.cross, state="hidden")
        self.c.coords(self.top, self.W / 2, self.H * 0.40)
        self.c.itemconfig(self.top, font=("Menlo", 24), justify="left")
        good = []
        while not self.space:
            x = board.get_current_board_data(2 * FS)
            lines, good, spreads = [], [], []
            for i, row in enumerate(EEG):
                v = x[row]; s = stuck(v)
                at = ((v == v.min()) | (v == v.max())).astype(int) if len(v) else np.zeros(1, int)
                flips = np.mean(np.abs(np.diff(at))) if len(at) > 1 else 0
                spreads.append(np.std(sosfiltfilt(sos, v - v.mean())) if len(v) > 3 * FS // 2 else 0)
                if s < 0.05:
                    good.append(i); note = "good"
                elif flips > 0.3:
                    note = "FLOATING: not touching the skin"
                else:
                    note = "no contact"
                lines.append("%s  %-20s %s" % ("✓" if s < 0.05 else "✗", NAMES[i], note))
            if len(good) >= 2:
                med = np.median([spreads[i] for i in good])
                for i in good:
                    if spreads[i] > 5 * med:
                        lines[i] += "   (very noisy: press it on)"
            self.text(top="SENSOR CONTACT\n\n" + "\n".join(lines) + "\n\nbest: ✓ on all four, especially the ears\nif you see ✗: adjust the headband, or slightly wet the sensors\n(or the skin and hair under them) with a little water",
                      bottom="SPACE when it looks good")
            self.wait(1)
        self.root.unbind("<space>")
        self.c.coords(self.top, self.W / 2, self.H * 0.18)
        self.c.itemconfig(self.top, font=("Helvetica", 32), justify="center")
        self.c.itemconfig(self.cross, state="normal")
        self.text(top="", bottom="")
        return good

    def close(self):
        try: self.root.destroy()
        except Exception: pass

# ---------------------------------------------------------------- session


def run():
    bank = load_bank()
    left_word, right_word = "RED", "BLUE"
    print("user %s: %d trials in the bank, %d marked with X" % (A.user, len(bank["labels"]), sum(bank["bad"])))
    params = BrainFlowInputParams()
    if A.name and not A.address:
        from find_muse import address_of
        A.address = address_of(A.name)
        print("%s -> %s" % (A.name, A.address))
    if A.address:
        params.mac_address = A.address
    params.timeout = 20
    board = BoardShim(BOARD, params)
    board.prepare_session(); board.start_stream()
    screen = None
    try:
        time.sleep(5)
        screen = Screen()
        channels = list(range(4)) if A.dry else screen.contact(board)
        if len(channels) < 2:
            raise SystemExit("fewer than two channels have contact; adjust the headband and try again")
        def mark_bad(event=None):
            if bank["bad"] and bank["sessions"][-1] == STAMP and not bank["bad"][-1]:
                bank["bad"][-1] = True
                save_bank(bank)
                screen.text(bottom="marked: the last trial will not be used for training")
                print("   last trial marked with X")
        screen.root.bind("<KeyPress-x>", mark_bad)

        def instructions(text, spoken=None):
            """Text stays on screen until SPACE (her remark: the first instruction flashed by too fast)."""
            screen.text(top=text, bottom="press SPACE when you are ready")
            if spoken:
                say(spoken)
            if A.dry:
                screen.wait(2)
            else:
                screen.wait_for_space()
            screen.text(top="", bottom="")

        instructions("RED = count backwards (for example 734, 727, 720...)\n"
                     "BLUE = recall a pleasant memory in detail\n\n"
                     "Keep your eyes on the cross, face and jaw relaxed, lips still.\n"
                     "Each trial: read the task, start it right away, keep going until STOP (8 seconds).\n"
                     "If you were not really in the state, press X right after the trial: it will not be used.\n"
                     "Q quits at any moment, everything is saved.",
                     "Red means counting backwards. Blue means recalling a pleasant memory. "
                     "Keep your eyes on the cross and your jaw relaxed.")

        def trial(side, live=None, kind="game"):
            screen.show(0)
            word = left_word if side == "left" else right_word
            text = task_text(side)
            screen.text(top="%s   %s" % (word, text), bottom="")
            say("%s. %s" % (word.capitalize(), text))
            t0, pos = time.time(), 0.0
            while time.time() - t0 < TRIAL:
                screen.wait(STEP)
                if live is None or time.time() - t0 < SKIP + 1.0:
                    continue
                n = int(min(time.time() - t0 - SKIP, WINDOW) * FS)
                raw = board.get_current_board_data(n + 2 * FS)[EEG]
                clean = remove_eyes(prep(raw), live["coefs"])[:, -n:]
                dec = live["model"].decision_function([features(clean, live["channels"]) - live.get("offset", 0)])[0]
                pos = float(np.clip(pos + GAIN * np.tanh(dec / live["scale"]), -1, 1))
                screen.show(pos)
            seg = board.get_current_board_data(SEG_N)[EEG]
            say("Stop.")                               # a clear end of every trial (as in Olga's recordings)
            hit = None
            if live is not None:
                hit = (pos > 0) == (side == "right")
                screen.text(bottom="hit" if hit else "miss")
            if seg.shape[1] >= SEG_N:
                bank["segs"].append(seg[:, :SEG_N]); bank["labels"].append(1 if side == "right" else 0)
                bank["sessions"].append(STAMP); bank["bad"].append(False)
                save_bank(bank)                                   # after every trial, so Q loses nothing
                if isinstance(live, dict) and isinstance(live.get("model"), Transferred):
                    live["model"].add(windows_of(seg[:, :SEG_N], live["coefs"], live["channels"]))
                elif isinstance(live, dict):
                    live["offset"] = session_offset(live, bank)
            screen.text(top="", bottom=("hit" if hit else "miss") + "      X = I was not in the state, do not use this trial" if live is not None
                        else "X = I was not in the state, do not use this trial")
            screen.wait(random.uniform(*PAUSE)); screen.text(bottom="")
            return hit

        def pause(label):
            screen.text(top=label, bottom="press SPACE to continue")
            say("Break. Press space when you are ready.")
            if A.dry:
                screen.wait(2)
            else:
                screen.wait_for_space()
                screen.contact(board)
            screen.text(top="", bottom="")

        def calibrate():
            instructions("CALIBRATION\n\n%d trials of 8 seconds, about 4 to 5 minutes.\n"
                         "The cross does not respond yet, just do the task shown." % CAL_TRIALS,
                         "Calibration. The cross does not respond yet, just do the task.")
            for side in shuffle_limited(["left", "right"] * (CAL_TRIALS // 2)):
                trial(side, kind="cal")
            pause("calibration done")

        def free_mode(live):
            instructions("FREE MODE\n\nPick a state yourself (red = counting, blue = remembering),\n"
                         "get into it first, and only then press SPACE.\n"
                         "The cross turns red or blue; when a color is reached the screen fills with it.\n"
                         "Q to finish.", "Free mode. Pick a state, get into it, and only then press space.")
            counts = {"left": 0, "right": 0, "none": 0}
            while True:
                screen.fill("black"); screen.show(0)
                screen.text(top="%s %d      %s %d" % (left_word, counts["left"], right_word, counts["right"]),
                            bottom="get into a state, then press SPACE")
                screen.wait_for_space()
                screen.text(bottom="")
                t0, pos, reached = time.time(), 0.0, None
                while time.time() - t0 < 20:
                    screen.wait(STEP)
                    n = int(WINDOW * FS)
                    raw = board.get_current_board_data(n + 2 * FS)[EEG]
                    clean = remove_eyes(prep(raw), live["coefs"])[:, -n:]
                    pos = float(np.clip(pos + GAIN * np.tanh(live["model"].decision_function([features(clean, live["channels"]) - live.get("offset", 0)])[0] / live["scale"]), -1, 1))
                    screen.show(pos)
                    if abs(pos) >= 1:
                        reached = "right" if pos > 0 else "left"
                        break
                counts[reached or "none"] += 1
                if reached:
                    screen.fill("#c81e1e" if reached == "left" else "#1e3cc8")
                    screen.text(top="", bottom="")
                    say("red" if reached == "left" else "blue")
                else:
                    say("not reached")
                screen.wait(2)

        transfer = None
        if A.free:
            if A.start_from:
                raise SystemExit("free mode uses your own model; play the game with --start-from first")
            if len(good(bank)[1]) < 12:
                raise SystemExit("Free mode needs a trained model: user %s has %d usable trials, at least 12 are needed.\n"
                                 "Run the calibration and the game first (the same command without --free)." % (A.user, len(good(bank)[1])))
            screen.text(top="training your model…")
            free_mode(train(*good(bank), channels))
        if A.start_from:
            # the transfer test: Olga's direction, your scaling; two warm-up trials first
            transfer = Transferred(A.start_from)
            while not A.dry and [c for c in transfer.channels if c not in channels]:
                need = ", ".join(NAMES[c].split()[0] for c in transfer.channels if c not in channels)
                instructions("The loaded model needs %s, and it has no contact now.\n"
                             "Adjust the headband or wet that sensor slightly, then press SPACE." % need,
                             "The loaded model needs %s. Please fix that sensor." % need)
                channels = screen.contact(board)
            instructions("TWO WARM-UP TRIALS\n\nthey measure your signal levels for the loaded model.\n"
                         "The cross does not respond yet, just do the task shown.",
                         "Two warm-up trials to measure your signal. The cross does not respond yet.")
            warm = []
            for side in shuffle_limited(["left", "right"]):
                trial(side, kind="warmup"); warm.append(bank["segs"][-1])
            transfer.coefs = fit_eye_coefs(warm)
            for s in warm:
                transfer.add(windows_of(s, transfer.coefs, transfer.channels))
        elif A.calibrate or len(good(bank)[1]) < 12:
            calibrate()

        for block in range(GAME_BLOCKS):
            if transfer is not None and len(good(bank)[1]) < 12:
                live = {"model": transfer, "coefs": transfer.coefs, "channels": transfer.channels, "scale": transfer.scale}
                print("block %d with the loaded model (its own held-out accuracy %.0f%% on %d trials)" % (block + 1, 100 * transfer.cv_acc, transfer.n))
                label = "Olga's model, no calibration"
            else:
                screen.text(top="training your model…")
                live = train(*good(bank), channels)
                pickle.dump(live, open("model_%s.pkl" % A.user, "wb"))
                print("your model: %d trials, held-out accuracy %.0f%%" % (live["n"], 100 * live["cv_acc"]))
                label = "your model, about %.0f%%" % (100 * live["cv_acc"])
            screen.text(top="block %d of %d" % (block + 1, GAME_BLOCKS), bottom=label)
            say("Block %d." % (block + 1)); screen.wait(3)
            hits = [trial(side, live) for side in shuffle_limited(["left", "right"] * (GAME_TRIALS // 2))]
            print("block %d: %d of %d" % (block + 1, sum(hits), len(hits)))
            with open("summary_%s.txt" % A.user, "a") as f:        # something to send back
                f.write("%s  board %s  block %d  %s  %d of %d hits\n" % (STAMP, A.board, block + 1, label, sum(hits), len(hits)))
            say("%d of %d." % (sum(hits), len(hits)))
            if block < GAME_BLOCKS - 1:
                pause("block %d: %d of %d" % (block + 1, sum(hits), len(hits)))
        screen.text(top="done", bottom=""); say("Done. Thank you.")
    except KeyboardInterrupt:
        print("\nstopped with Q, saving")
    except BaseException as e:
        msg = str(e) if isinstance(e, SystemExit) else "%s: %s" % (type(e).__name__, e)
        with open("two_states_last_error.txt", "w") as f:
            f.write("%s  user %s  board %s\n%s\n\n%s" % (STAMP, A.user, A.board, msg, traceback.format_exc()))
        print("\nSTOPPED:", msg, "\n(details in two_states_last_error.txt)")
        if screen is not None:
            try:
                screen.stopped = False
                screen.text(top="STOPPED\n\n" + msg, bottom="closing in 8 seconds (details in two_states_last_error.txt)")
                screen.wait(8)
            except BaseException:
                pass
    finally:
        if screen is not None:
            screen.close()
        save_bank(bank)
        try:
            DataFilter.write_file(board.get_board_data(), "raw_%s_%s.csv" % (A.user, STAMP), "w")
        except Exception as e:
            print("could not save raw data:", e)
        print("saved: %s (%d trials, %d marked with X) and raw_%s_%s.csv" % (BANK, len(bank["labels"]), sum(bank["bad"]), A.user, STAMP))
        try:
            board.stop_stream(); board.release_session()
        except Exception:
            pass


if __name__ == "__main__":
    run()
