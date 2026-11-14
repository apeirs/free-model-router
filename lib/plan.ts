// Which model serves the next request, and what is known about each provider's budget.
//
// A provider's free-tier budget is drawn per trial and never published, but its shape is not a secret: the
// first call a provider takes is never refused for its schedule, a budget holds at least `min` calls, and no
// more than `max`. Scheduled refusals are one or two calls among the second to the one before the last.
// The ledger below turns what this server has seen (calls made, refusals met) into the chance that the next
// call to a provider is refused, and `pick` trades that chance against the rating a model would lose.
//
// Pure: no clock, no randomness.

export interface Span {
  min: number;
  max: number;
}

/** The range of each provider's drawn call budget, by provider id. */
export const BUDGETS: Record<string, Span> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};
const UNKNOWN: Span = { min: 4, max: 16 };

/** What one refused call costs, in weighted rating points: the price of a risk, set from simulated trials. */
export const RISK_PRICE = 150;

export class Ledger {
  /** Calls that counted against a provider's budget, refusals included. */
  readonly made = new Map<string, number>();
  /** Refusals met. */
  readonly refused = new Map<string, number>();
  /** Whether the provider's latest call was refused. */
  readonly lastRefused = new Map<string, boolean>();
  readonly gone = new Set<string>();

  record(provider: string, ok: boolean): void {
    const made = (this.made.get(provider) ?? 0) + 1;
    this.made.set(provider, made);
    this.lastRefused.set(provider, !ok);
    this.streak.set(provider, ok ? 0 : (this.streak.get(provider) ?? 0) + 1);
    if (!ok) this.refused.set(provider, (this.refused.get(provider) ?? 0) + 1);
    if (!ok && (made >= (BUDGETS[provider] ?? UNKNOWN).max || (this.streak.get(provider) ?? 0) >= 2)) this.gone.add(provider);
  }

  private readonly streak = new Map<string, number>();

  /** The chance that the next call to a provider is refused; 1 when it is known to be out. */
  hazard(provider: string): number {
    if (this.gone.has(provider)) return 1;
    const span = BUDGETS[provider] ?? UNKNOWN;
    const n = this.made.get(provider) ?? 0;
    const next = n + 1;
    if (next > span.max) return 1;
    const exhausted = next <= span.min ? 0 : (next - span.min) / (span.max - span.min + 1);
    let scheduled = 0;
    if (next >= 2) {
      const remaining = Math.max(0, 1.6 - (this.refused.get(provider) ?? 0));
      const slots = Math.max(1, (span.min + span.max) / 2 - 1 - n);
      scheduled = Math.min(1, remaining / slots);
    }
    let h = 1 - (1 - exhausted) * (1 - scheduled);
    if (this.lastRefused.get(provider) === true) h = Math.max(h, next > span.min ? 0.6 : 0.25);
    return h;
  }
}

export interface Choice {
  provider: string;
  rating: number;
}

/**
 * The index of the candidate to call: the least weighted rating lost against `best`, plus the risk price
 * times the chance of a refusal. Ties go to the earlier candidate (the list is best-rated first). -1 when
 * no candidate is left.
 */
export function pick(candidates: readonly Choice[], ledger: Ledger, best: number, weight: number, price = RISK_PRICE): number {
  let at = -1;
  let cost = Infinity;
  for (const [i, c] of candidates.entries()) {
    const h = ledger.hazard(c.provider);
    const total = weight * Math.max(0, best - c.rating) + price * h;
    if (total < cost - 1e-9) {
      cost = total;
      at = i;
    }
  }
  return at;
}
