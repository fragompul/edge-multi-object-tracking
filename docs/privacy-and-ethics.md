# Privacy & ethics

People tracking is dual-use. This project is built so that the safe path is the default one.

## Design choices

- **Session-local, anonymous IDs.** A track ID is an integer valid within one video. Nothing links IDs across videos,
  cameras or sessions, and no identity (name, face, biometric template) is ever produced.
- **Appearance embeddings are working memory.** ReID features exist only to keep identities straight *within* a clip,
  live in RAM, and are not persisted by the pipeline. (The benchmark caches embeddings of a *public research dataset*.)
- **The browser demo uploads nothing.** Webcam and file inputs are processed locally; there is no server to receive them.
- **Analytics use geometry only** (boxes, foot points, timing), never appearance.
- **Honest about what an ID is.** The agent is instructed to say that "unique people" is an upper bound.

## Out of scope on purpose

Cross-camera or cross-session re-identification, face recognition, attribute inference (age, gender, emotion),
and watch-lists. Persistent re-identification turns a counting tool into a tracking-of-individuals tool and, in the EU,
engages biometric-data rules (GDPR Art. 9) and, depending on the use, the AI Act's provisions on remote biometric
identification and emotion recognition. This is general information, not legal advice -- deployments need their own
legal review (purpose limitation, data minimisation, retention, signage, DPIA).

## Using it responsibly

- Prefer **aggregate outputs** (counts, flows, dwell distributions) over per-person trajectories; do not store trajectories
  longer than needed. Blur or drop frames at the edge and ship only counts.
- Evaluate **error rates before acting on outputs**: HOTA ~0.4 on hard footage means identity errors are routine.
  Never use tracker output as the sole basis for a decision about an individual.
- MOT17 contains real people filmed in public; its licence (CC BY-NC-SA 3.0) restricts commercial use.
