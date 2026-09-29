# Priors

Durable lessons about what works, baked in so every project starts with them instead of
rediscovering them. Read at song-caption time (stage 1) and scene-plan time (stage 4); add a new
section when something is learned that should outlive one project. Keep each prior short, sourced,
and actionable. Started 2026-09-28 with the two below.

## The grammar of music videos (Tom Scott, 2018)

"If Educational Videos Were Filmed Like Music Videos", 3:05, ~17M views, dir. Sammy Paul, feat.
dodie: https://youtu.be/G025oxyWv0E . A real director applies every music-video device to a
talking-head explainer while Scott narrates what each device is doing and why it is there. We make
educational videos in the music-video grammar, so this is the checklist of moves, and of what must
not be allowed to eat comprehension:

- Off-speed playback: filmed 30 fps, played at 24 (80% speed), which is where music video's
  "slightly unreal" feel comes from. Ours: subtle speed ramps inside one continuous scene (an
  allowed course transform), never a surprise pop; the gate counts frame-pops as hard changes.
- Rapid cuts with no continuity ("different clothes, different locations, different everything"):
  it works because the song is the through-line and the picture owes the viewer nothing. Ours, fast
  lane: exactly this grammar, cut on the beat, owe nothing. Ours, course: the sung lyric is the
  through-line, so continuity is the rule and cuts are rationed.
- Extreme slow-mo cutaways: "looks spectacular", lets a moody stare be held "without having to hold
  it for so long that it becomes awkward". Ours: slow motion is how a held number stays alive while
  the lyric finishes its line (course: one number held until the lyric moves on, nothing frozen
  > 3 s).
- Sped-up playback: dancers "seem more synchronized and precise than they really are", movements
  look "superhuman". Ours: rendered motion with sub-frame sampling and shutter blur reads as
  practiced rather than robotic; use it on builds and drops.
- A wind machine (a hardware-store leaf blower, receipt kept): manufactured drama in a single
  shot. Ours: the code equivalent is free; spend it as the one added layer on a chorus return.
- Lip-sync to a prerecorded vocal, and "if my lips don't quite match up at some point we'll just
  cut to another tape or to a completely random slow-mo shot of an object being destroyed and no
  one will care". Ours: this is the beat-cut grammar we already run on; the cut forgives sync
  error. The alignment data (word times) is the prerecorded vocal; scenes are the tapes.
- The making-of is faked too: the "studio" is a different room, the mic is unplugged, the featured
  artist filmed separately while the video insists everyone is in the same room. Viewers accept
  presentation over provenance. Ours: the opposite bargain, real data only, computed numbers
  labelled; the credibility is the channel.
- The middle eight by a featured artist exists "in an attempt to cross-promote". Ours: the 4-8
  instrumental bars after a new idea (course rule) are our middle eight; the picture finishes the
  thought alone.
- A diegetic-audio break, inserted "just so people can't rip the whole thing". Ours: we want to be
  quoted, but a diegetic break (the model's real sampled text, a terminal beep, fan noise under a
  big render) is a strong section divider.
- The fake party: "it may look like we're having a spectacular time, as that's the image we want
  to project, but in reality we are on our fifteenth take". Ours: project energy through the
  arrangement, never claim it.
- Circular narrative: "ending the same way we started makes the audience think that there's a
  sensible circular narrative that ties everything together even if there isn't". Ours: open and
  close each video on the same image; the last chorus reassembles every piece and the outro names
  the next episode, so ours is real, not implied.
- It is a promo, not a song ("you can't buy this on iTunes"): the video sells something else.
  Ours: inverted, the song is the product and the teaching is the hook; when a device serves promo
  grammar (spectacle for its own sake) over the one idea being taught, cut the device.

## Four chords (the default harmony)

The four-chord loop, the roots of I, V, vi and IV in any order (the "Axis of Awesome" family,
hundreds of hits), is pop's most familiar harmony: listeners parse it instantly, it recycles for a
whole song without wearing out, and it makes a chorus feel inevitable. Default for every song here
unless the genre genuinely refuses it:

- Name the harmony in every run caption, alongside genre, BPM and vocal. Before 2026-09-28 no
  caption ever named chords; they should now. Phrasing that works: "built on a repeating I-V-vi-IV
  four-chord loop", "Motown I-vi-IV-V doo-wop changes under the call and response".
- The chorus states the loop plainly and every chorus return uses the same order. The chorus is
  "same words, same picture", so give it the same four chords too. Verses may rotate the order
  (vi-IV-I-V) for movement; the bridge may leave the loop so the final return lands harder.
- Verify takes with `uv run tools/song/musical.py <take.wav> ...`: the `4chords` column reports the
  share of beats on the loop's four roots and the order it actually cycles in. Prefer takes whose
  chorus sits on the loop. This guides picking; it is not a gate (the pace rules gate, this
  steers).
