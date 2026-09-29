# Scene-author brief template

One brief per scene, filled by the lead and sent verbatim to the scene author (a Claude Code or
GLM worker agent). XML-wrapped per tools/prompts.md: Claude-side briefs carry the richest tagging.
Every {TOKEN} is filled before sending; delete any section that has nothing to say rather than
leaving it empty.

<task>
Author {SCENE_FILE} for {PROJECT} ("{SONG_TITLE}", episode {EPISODE} of ML, slowly):
{SECTION_NAME}, lyric lines "{FIRST_LINE}" through "{LAST_LINE}" (lines {N}-{M} of the lyric
sheet). The scene covers exactly these lines; the neighbouring scenes are {PREV_SCENE} before and
{NEXT_SCENE} after, so do not visually repeat their key image ({PREV_IMAGE} / {NEXT_IMAGE}).
You own {SCENE_FILE} only; every other file is read-only. Kit, engine, overlay and timeline
changes go through the lead: put them in your report instead of editing.
</task>

<rules>
{RULES_SUMMARY: for course episodes, the pace rules that bind this section (one idea per scene,
hard-change budget per minute, nothing frozen > 3 s, one number at a time held until the lyric
moves on, the sung lyric line stays on screen, 4-8 instrumental bars = the picture finishing the
idea); for fast-lane videos, the beat-cut grammar instead. Also: the episode's colour meanings
(pink = the one thing being taught, blue = the data, ink = structure) and which devices from
tools/priors.md this scene may use.}
</rules>

<windows>
Section {SECTION_NAME} runs {START_TIME}-{END_TIME}. Cut windows (beat-aligned, from the shot
plan): {WINDOWS_LIST}. Word-level times come from {TIMES_SOURCE: data/sections.json or
data/words.json}; the lyric line must be on screen across {LYRIC_LINE_SPAN}.
</windows>

<real_data>
{FACTS: every number or string this scene may put on screen, with its source and file
(data/*.json), plus the exact label for any computed value. Real numbers only; if a number is
not in this list, it does not go on screen.}
</real_data>

<deliverables>
1. {SCENE_FILE}, implementing the plan for these lines only (kit types and helper names as in
   {REFERENCE_SCENE}).
2. A contact sheet proving it: {CONTACT_SHEET_COMMAND}. Send the PNG path back; do not merge,
   commit, render or touch the timeline.
</deliverables>
