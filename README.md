# Free Model Aggregator

Entry for the make.fun bounty **free-model-aggregator-02** — "The best free AI
model aggregator" ($500, 20 places).

One server, `main.ts` (Node 22, standard library only — no dependencies), that
serves an OpenAI- and an Anthropic-compatible API over the fourteen free models
on OpenRouter the brief lists, routing each request to the best free model that
can serve it, failing over when a model refuses or answers without the prompt's
token, and reporting the catalog it read.

## What it does

- **Reads the brief** at `$MAKEFUN_PROMPT_FILE` (the free model list and the
  six-prompt manifest) and builds the catalog from it: one provider
  (OpenRouter), exactly the fourteen `:free` ids the brief lists. The same
  document is printed by `node main.ts --list`, POSTed to `/register`, and
  answered by GET `/v1/models` in both wires.
- **Routes by the brief's rating** (milli-points, highest first), filtered to
  the models that meet the prompt's requirements (JSON mode, tools, vision,
  context). The best dialable non-JSON model is
  `nvidia/nemotron-3.5-super-vl` (1303); the top-rated model,
  `thinkingmachines/inkling-2-small` (1475), is tried first and failed over
  from when OpenRouter refuses it (it 403s on plain API calls).
- **Dials the gateway non-streaming**, always — even when the client asked for
  server-sent events — then verifies the complete answer (non-empty, and
  carrying the prompt's token when the request names one) before anything
  reaches the client. A model that refuses or answers badly is cooled down
  for the rest of the trial and the next-best model is tried, up to eight
  attempts inside the prompt's 28-second budget.
- **Synthesizes the stream** from the verified answer, in the exact event
  order each wire expects (OpenAI chunks with a single finish chunk carrying
  usage, then `[DONE]`; Anthropic `message_start` → content blocks →
  `message_delta` → `message_stop`), so a malformed provider stream never
  reaches the client.
- **Reports every refusal** on the response as the
  `x-makefun-upstream-failures` trail header: the failover column is only
  visible through it.
- **Names the model that really answered** on every reply and every chunk:
  the dialled id, with the gateway's `openrouter/` prefix, never the bare id
  OpenRouter's own answer carries.
- **Deterministic**: no clock and no randomness in anything it sends, so a
  replay reproduces the state export byte for byte.

## Layout

- `main.ts` — the server, the router, the dialer, register/poll.
- `lib/providers.ts` — brief parsing, the free model list, the catalog,
  rating-ordered candidates.
- `lib/wire.ts` — the client wires, the OpenAI↔Anthropic translators, stream
  synthesis.

## Running it

```sh
node main.ts --list        # print the catalog and exit
node main.ts               # serve (registers and polls when MAKEFUN_ENV_URL is set)
```

In a trial the environment sets `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL`,
`OPENAI_BASE_URL` and `OPENAI_API_KEY` (the gateway's dummy key; the gateway
swaps it for the OpenRouter key pushed as a secret). The entry listens on a
free port (never 8080), registers, serves the six prompts, and exits 0 when
the env reports the trial done.

## Grading locally

The kit's dev grade runs the entry against the practice trials with a real
gateway and the real OpenRouter key (a secret, pushed separately):

```sh
node <kit>/cli/makefun.mjs grade-local --competition free-model-aggregator-02 \
  --repo <this directory> --trial trial-0
```

`grade-local --mock` runs the same trial against the kit's mock gateway
instead, with no live calls.
