# Prompt wrapping

Every prompt this repo hands to a model is wrapped in lowercase XML tags, one section per tag.
Wrapping researched 2026-09-28: Anthropic's prompting guidance has Claude parse mixed
instructions/context unambiguously through XML tags; OpenAI's guide names XML tags as a
first-class delimiter alongside section headings. The depth differs by audience:

- **Claude** (scene-author briefs, agent notes, GLM worker briefs): richest tagging; use as many
  tags as the brief has sections. Template: `tools/scene_brief_template.md`
  (task, rules, windows, real_data, deliverables).
- **Codex / GPT** (`tools/publish/studio_prompts/*.md`): exactly four tags, because reasoning
  models prefer leaner prompts: `<context>` (who, where, which tool), `<constraints>` (the hard
  limits and never-rules), `<task>` (numbered steps plus all variable data: video ids, captions,
  schedules, file paths), `<report>` (the required report format). Placeholders like `{VIDEO_ID}`
  and `{TEXT}` are substituted by plain string replace (`release_comment.sh`), so keep tokens
  verbatim inside the tags.
- **ACE-Step song captions: the one deliberate exception.** The 4B planning LM was trained on
  natural tag-style captions; wrapping those in XML is distribution shift, so captions stay plain
  comma-separated tags (see `tools/priors.md` for what they must name).

Tags mark boundaries the model must not bleed across (instructions vs site data vs report
format); they do not replace the numbered steps, exact strings and hard limits inside.
