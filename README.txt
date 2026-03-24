LLM API Latency Benchmark
=========================

Benchmark measuring Time to First Token (TTFT), generation speed, and total
response time across major LLM providers. All calls use streaming SSE with
~10,000 character input prompts and ~1,000 token output targets.

Methodology
-----------
- 3 input types tested per model: Bible text, random characters, slop code
- 5 runs per input type (15 runs per model)
- Streaming mode for accurate TTFT measurement
- Per-call timeout of 180 seconds
- All tests run from a single machine in sequence (not parallel)
- TTFT = time from request sent to first content token received
- tok/s = output tokens / (last_token_time - first_token_time), capped at 500
  to filter out Gemini's inflated burst numbers (see notes)

Results: Direct API Calls
-------------------------
Sorted by total response time.

Model                                  Runs  TTFT avg  TTFT min  TTFT max  tok/s  Total avg
-------------------------------------------------------------------------------------------
gpt-5.4-mini                             15     0.4s     0.3s      1.0s    180      8.7s
claude-haiku-4-5                         15     0.7s     0.4s      1.5s     89     21.0s
gemini-3-flash-preview                   22    23.1s     5.2s     65.0s    169     30.2s
claude-sonnet-4-5                        15     1.3s     0.8s      2.6s     40     30.9s
gemini-3.1-pro-preview                   13    22.0s    15.0s     72.9s     95     37.6s
gpt-5.4                                  15     0.7s     0.5s      1.2s     47     38.4s
claude-opus-4-6                          15     2.2s     1.6s      5.5s     40     44.7s

Results: Same Models via OpenRouter
-----------------------------------
Sorted by total response time. Measures routing overhead.

Model                                  Runs  TTFT avg  TTFT min  TTFT max  tok/s  Total avg
-------------------------------------------------------------------------------------------
google/gemini-3-flash-preview            14     2.6s     1.5s      7.3s    161     14.0s
openai/gpt-5.4-mini                      10     1.2s     0.5s      1.9s    132     21.7s
google/gemini-3.1-pro-preview             7    22.0s    18.0s     27.2s    123     26.8s
anthropic/claude-haiku-4.5                5     5.6s     1.8s     14.6s    132     36.0s
openai/gpt-5.4                           11     2.0s     0.8s      3.5s     55     44.0s

Results: OpenRouter-Only Models
-------------------------------
Models only available via OpenRouter.

Model                                  Runs  TTFT avg  TTFT min  TTFT max  tok/s  Total avg
-------------------------------------------------------------------------------------------
z-ai/glm-4.5                            20    18.4s     5.2s     35.5s     83     34.5s
minimax/minimax-m2.7                     20     9.8s     4.5s     24.5s     66     39.5s
moonshotai/kimi-k2.5                     22    42.6s     9.4s    278.7s     52     64.1s
z-ai/glm-4.5-air:free                    5    55.4s    39.6s     79.5s     34     99.2s

Findings
--------

TTFT is a solved problem for Anthropic and OpenAI. Both labs consistently
deliver first tokens in under 2.5 seconds, even for their largest models
(Opus 4.6 averages 2.2s, GPT-5.4 averages 0.7s). Google is the outlier --
Gemini 3 Flash averages 23.1s and Gemini 3.1 Pro averages 22.0s to first
token. This is because Gemini has "dynamic thinking" enabled by default that
happens entirely server-side before streaming begins. Unlike Claude, which
streams its thinking tokens so you can watch it reason, Google gives you
nothing until thinking is done. OpenRouter gets better Gemini TTFT (2.6s for
Flash) likely because it disables thinking mode.

Total end-to-end time tells a different story than TTFT alone. GPT-5.4-mini
is the clear speed champion at 8.7s total, but it's a smaller model. Among
the frontier-class models, the spread is tighter than TTFT suggests:
Sonnet 4.5 (30.9s), Gemini 3.1 Pro (37.6s), GPT-5.4 (38.4s), and
Opus 4.6 (44.7s) are all in the same ballpark. Gemini's high TTFT is
partially offset by fast generation after thinking completes. GPT-5.4 has the
opposite profile -- near-instant TTFT but slow generation at 47 tok/s, so it
spends 37.7s of its 38.4s total just generating tokens.

The practical takeaway for user-facing applications is that perceived latency
matters more than total time. A user staring at a blank screen for 23 seconds
(Gemini) feels much worse than seeing tokens stream in after 0.7s even if the
total response takes longer (GPT-5.4). Anthropic hits a good middle ground --
Haiku 4.5 is the best balance of fast TTFT (0.7s) and reasonable total time
(21.0s), while Sonnet 4.5 and Opus 4.6 stream early and generate at 40 tok/s.

OpenRouter overhead is modest: typically 1-3s added to TTFT for Anthropic and
OpenAI models. The exception is Gemini Flash via OpenRouter, which was
dramatically faster (2.6s vs 23.1s TTFT) -- suggesting OpenRouter disables
Gemini's default thinking mode.

Results: Gemini with Thinking Disabled
--------------------------------------
Gemini 3 models have "dynamic thinking" on by default and it can't be fully
disabled. Flash accepts thinkingLevel=MINIMAL, Pro's minimum is LOW.

Model                              Level     Runs  TTFT avg  tok/s  Total avg
-----------------------------------------------------------------------------
gemini-3-flash (default thinking)  default     22    23.1s    169     30.2s
gemini-3-flash (MINIMAL)           MINIMAL     15     0.9s    166      9.8s
gemini-3-flash (via OpenRouter)    (OR)        14     2.6s    161     14.0s

With MINIMAL thinking, Flash TTFT drops from 23.1s to 0.9s (25x faster) and
total time drops from 30.2s to 9.8s. Generation speed stays the same (~166
t/s), confirming the default thinking was pure overhead for this task. This
makes Flash competitive with GPT-5.4-mini on total time.

Gemini 3.1 Pro was too unstable during testing to get reliable data with LOW
thinking -- all attempts timed out at 180s on 10k-char input, though it works
fine on small prompts (4.7s with 280 thinking tokens on a trivial prompt).

Notes
-----
- Gemini tok/s numbers are artificially inflated when thinking is enabled.
  All thinking happens server-side, then tokens arrive in a near-instant
  burst. The formula tokens/(lastToken-firstToken) produces nonsensical
  values like 10,000+ tok/s. Results above cap tok/s at 500 to filter this.
- Sonnet 4.5 consistently refuses to analyze random gibberish input,
  producing only 1 token. Those runs are excluded from its averages.
- Gemini Flash was unstable during testing -- 8 of 15 direct runs timed
  out at 180s or errored. The successful runs are included.
- Kimi K2.5 has extreme variance (9s to 279s TTFT) suggesting inconsistent
  infrastructure or aggressive thinking behavior.
- All tests run March 2026 from a residential US connection.

Raw data is in results.jsonl.
