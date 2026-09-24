# Right of Way

Fencelytics uses a **temporal rule-based state machine** to combine action recognition, movement, and scoreboard events into a right-of-way interpretation for foil fencing.

Right-of-way is not determined from a single frame. The system considers the sequence and timing of movement and fencing actions, then uses the electronic scoreboard to determine when a touch has been registered.

## Inputs

The right-of-way system combines information from several stages of the pipeline:

* **Action recognition** — identifies upper- and lower-body actions such as Extend, Parry, Lunge, and Recover.
* **Motion tracking** — determines movement along the piste.
* **Scoreboard detection** — identifies scoring events from the electronic scoreboard.

These signals are processed temporally rather than independently frame by frame.

## Scoreboard Detection

The scoreboard system uses a two-stage approach.

### 1. Scoreboard localization

The YOLOv7 object detector identifies the **Scoreboard** region in each frame. If multiple scoreboard detections are present, the system selects the largest detected scoreboard region rather than simply choosing the highest-confidence detection. This helps avoid a smaller spurious detection replacing the main scoreboard.

The detected region is then cropped from the frame.

### 2. Light detection

The cropped scoreboard region is analyzed using **HSV color thresholds** to count illuminated red and green pixels.

* **Red light → LEFT fencer**
* **Green light → RIGHT fencer**

The system counts colored pixels across the **entire scoreboard box** rather than assuming that the lights occupy fixed positions. This makes the detector less dependent on a particular scoreboard layout.

### Adaptive baseline and debouncing

Simply detecting red or green pixels is not sufficient because scoreboard regions may contain colored pixels even when a light is not active. Fencelytics therefore maintains a separate **adaptive exponential moving average (EMA) baseline** for the red and green pixel counts.

A light is considered active only when its current pixel count satisfies both:

* A minimum ratio above its adaptive baseline
* A minimum absolute increase above the baseline

The baseline is updated only while a light is considered **off**, preventing an illuminated light from gradually raising its own baseline.

A light must then remain continuously active for a minimum number of frames before it is confirmed as a scoring event.

### Double-touch detection

The red and green detections share a single debouncing process.

When one light is confirmed, the system briefly waits for the other light before reporting the event. This is necessary because the two lights in a genuine double touch may be confirmed a few frames apart rather than on exactly the same frame.

If both lights are confirmed within the settle window, the event is recorded as a **double touch**. Otherwise, the confirmed light is recorded as a single touch.

After a scoring event, the detector enters a short lockout period to prevent the same illuminated light from being counted repeatedly.

This separates **light detection** from **touch reporting**, allowing the system to distinguish a genuine scoring event from individual noisy frames.

## Right-of-Way State Machine

Movement is used to establish and maintain the current right-of-way state:

* Both fencers advancing → **Neither**
* Only LEFT advancing → **LEFT**
* Only RIGHT advancing → **RIGHT**
* Neither advancing → current right-of-way remains unchanged

Once a fencer has established right-of-way, it remains locked while that fencer continues advancing. The holder only gives up right-of-way after they stop advancing for the configured grace period.

When both fencers are advancing, an **Extend + Lunge overlap** can act as a tie-break. The system identifies an attack only when the upper-body Extend and lower-body Lunge signals are simultaneously active.

When a scoreboard event occurs:

* **Single light:** the corresponding fencer scores directly.
* **Double touch:** the current right-of-way holder receives the point if an established holder exists.
* **Unresolved situation:** the system does not guess when the available information is insufficient.

## Output

The system produces:

* A referee log containing right-of-way transitions, attacks, scoring events, and the final score.
* An annotated video showing the right-of-way state and running score.
* A per-fencer action log showing movement, upper-body actions, lower-body actions, and detected attacks.

Example logs are provided in:

```text
examples/
```

## Limitations

The current right-of-way system is a prototype and does not model every possible foil exchange.

In particular, **parry-riposte sequences are not currently modeled comprehensively**. When a touch cannot be resolved from the current movement, action, and scoreboard information, the system reports it as unresolved rather than making an unsupported decision.

Scoreboard detection can also be affected by poor visibility of the scoreboard, unusual scoreboard layouts, camera movement, lighting, or inaccurate YOLO scoreboard localization.
