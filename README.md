# free-model-aggregator-01-entry

The best free AI model aggregator — one server that serves the OpenAI Chat
Completions API and the Anthropic Messages API over ten simulated free-tier
model providers, in JSON and in server-sent events.

Owner: **sofyan-adi** · referral: `yx3n09hc`

## What it does

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
  sealed quality table and the workload manifest.
- Probes each provider's catalog route in the provider's own wire (OpenAI,
  Anthropic or Gemini) and builds one catalog.
- For each request, picks the best **free** model that meets the request's
  requirements: highest rating first, then fewest refusals, then fewest calls,
  then model id. JSON-mode requests never go to an anthropic-wire provider.
- Fails over on a refusal (402, 429, 5xx or a timeout): one provider per
  attempt, at most eight attempts. A 402, a spent-budget or outage message, or
  a third refusal spends the provider for the rest of the trial. A 2xx answer
  is never retried.
- Translates requests and answers between the OpenAI, Anthropic and Gemini
  wires, keeping the system prompt, user text, images, tools, stop sequences,
  max_tokens, JSON mode, usage and stop reason exactly as sent.
- Registers its port and catalog with the env, serves the twenty requests,
  polls `GET /work` every 100 ms and exits 0 when the workload is done.

## Files

- `main.ts` — the entrypoint: brief, probe, catalog, routes, register, poll.
- `lib/providers.ts` — the brief's parser, the catalog probe, the router's books.
- `lib/wire.ts` — the OpenAI / Anthropic / Gemini translations and renderers.

## Run it

Node 22, standard library only, nothing to install:

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
```

Outside a trial, with neither variable set, `--list` prints
`{"providers":[]}` and exits 0.

## License

MIT — see `LICENSE`.
