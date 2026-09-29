<context>
You are operating Google Chrome on this Mac. Elliot is signed in to TikTok for Developers (developers.tiktok.com,
open in a tab) and to TikTok. Use computer use / the Chrome browser tool. Goal: a developer app that
lets a local CLI upload videos to the signed-in account's TikTok drafts (Elliot finishes posting in the app).
</context>

<constraints>
Hard limits: do not sign in, sign up, type or reveal any password, enter verification codes, or accept any terms,
policies or agreements (if saving needs a terms checkbox, stop and quote its text). Do not submit the app for
review. Do not post, follow, like or message anything on TikTok. Never print the client secret in your report.
</constraints>

<task>
1. Confirm the developer portal is signed in (Manage apps visible). If it is not, stop and say so.
2. Manage apps -> Connect an app (create app). Use:
   - App name: ai-video
   - App icon: out/tiktok/app_icon.png
   - Category: Education
   - Description: "Uploads Elliot Arledge's own code-rendered educational music videos (how GPUs and LLMs work) from
     his computer to his TikTok drafts, where he reviews and posts them."
   - Terms of Service URL / Privacy Policy URL: leave empty if the form allows saving without them; otherwise stop
     and report that they are required.
   - Platform: Desktop if offered (the CLI runs on a Mac); otherwise Web with website https://www.youtube.com/@elliotarledge
3. Add products: Login Kit and Content Posting API. Scopes: user.info.basic and video.upload only (no
   video.publish; do not enable "Direct Post"). Login Kit redirect URI: http://localhost:8765/callback (if the form
   rejects http/localhost for this platform, report the exact validation message and try
   https://localhost:8765/callback).
4. Save (not submit). If the app has a Sandbox option, create a sandbox for it, apply the same products, scopes and
   redirect URI, and add the signed-in account as a target user. Report what the sandbox page says is needed to finish
   adding the target user.
5. Credentials: for the sandbox (and the production app if shown), write the Client key and Client secret into
   ~/.config/tiktok/client.json as {"sandbox": {"client_key": "...", "client_secret": "..."},
   "production": {...}} with file mode 600 (create the folder with mode 700). Do not print them.

File paths above are relative to the repo root (Codex runs with `-C <repo>`).
</task>

<report>
Report in short lines: app id or name as shown, status, platform, products, scopes, redirect URI accepted or the
exact error, sandbox and target user state, what is still missing, and whether client.json was written.
</report>
