# free-model-aggregator-01-entry

One server (`main.ts`, Node 22, standard library only) that serves the OpenAI
Chat Completions API and the Anthropic Messages API over ten simulated
free-tier model providers, in JSON and in server-sent events.

- Reads the trial brief (provider directory, sealed quality table, workload
  manifest), probes every provider's catalog route in its own wire (OpenAI,
  Anthropic, Gemini), and registers one catalog with the env.
- Answers each of the 20 client requests with the highest-rated free model
  meeting that request's requirements, failing over to the next-best free
  model on any 402/429/5xx, respecting one-call-at-a-time per provider.
- Translates natively between all six client/provider wire pairs so the
  provider's echo of the system prompt, user text, images, tools,
  `max_tokens`, stop sequences and JSON mode matches what the client sent;
  returns the provider's text whole, its usage counts as reported, and its
  stop reason mapped to the same family.
- `node main.ts --list` prints the registered catalog, byte for byte.
- Deterministic: no wall clock, no unseeded randomness — a replay leaves the
  same state export byte for byte.

Local validation against the kit's sealed env + scorer (Docker-free, in
process): practice trials 0, 1 and 2 all score **99** (quality 99, served
100, wire 100, failover 100, catalog 100), 20/20 requests valid, 233/233
response checks passed; two live runs of trial-0 leave byte-identical state
exports. A starved-budget run (every provider but Gemini dead) still scores
99 with 20/20 valid, exercising the Anthropic→Gemini path.
