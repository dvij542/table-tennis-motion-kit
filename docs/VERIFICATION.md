# Standalone Handoff Verification

Verified November 2, 2026 on Linux x86_64, Python 3.11.15, using the committed
uv.lock. A fresh clone in a different temporary directory was given its own
virtual environment. It did not import the original project or load its assets.

- 15 tests passed in the original standalone checkout and in the relocated clone.
- Integrity checks covered all 38 copied asset files and all saved example files.
- Recomputed FK for all 20 motions / 2,520 frames: maximum contact-path error
  0.0018802292 m; maximum orientation error 0.43130531 degrees. Frozen pelvis,
  legs and left-arm joints, joint bounds, target grid, loop endpoints and
  axis-angle serialization passed.
- Fresh nominal generation passed: 1.5696810 mm path error, 0.22552923 degrees
  orientation error, 24 sequence-refinement evaluations.
- Fresh farthest-right/lowest generation (Y offset -0.15 m, Z offset 0) passed:
  1.8802199 mm path error, 0.43130531 degrees, 119 refinement evaluations.
- Fresh generation was tested for those TWO targets, not all 20. Existing numeric
  examples were preserved; all 20 were independently checked with FK.
- Browser test against the relocated clone loaded the real robot/table/racket,
  selected a different Y/Z target, verified the rendered image changed while
  preserving paused frame 50, and verified mouse orbit. Desktop and mobile
  screenshots rendered, with no JavaScript errors. docs/preview.png is the
  desktop impact screenshot from that test.
- Optional video renderer produced H.264, 1280x800, 25 fps, 7.56 s (one 1x and
  one 0.5x replay). Video generation used the relocated clone and its own deps.
- Only the temporary verification viewer was stopped. The original user's
  viewer/service was not modified.

Not tested: every target freshly reoptimized in the standalone environment,
other operating systems, dynamic tracking, balance, self-collision, physical
ball contact, or real robot execution. No hardware safety claim is made.
