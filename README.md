# free-model-aggregator

One Node 22 server (`main.ts`, standard library only) that fronts ten free-tier model providers.

It serves the OpenAI Chat Completions API (`POST /v1/chat/completions`), the Anthropic Messages API
(`POST /v1/messages`), both as JSON or server-sent events, and `GET /v1/models`. For each request it picks the
highest-rated free model that meets the request's needs (tools, images, JSON mode, context length), speaks that
provider's own wire (OpenAI, Anthropic or Gemini), translates the answer back to the client's wire, and fails
over to the next model when a provider answers 402 or 429.

Run: `node main.ts` inside a trial (it reads `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`, registers, and exits 0
when the env reports the workload done). `node main.ts --list` prints the catalog document it registers.
