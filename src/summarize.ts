import * as fs from "fs";
import * as path from "path";

interface TimingResult {
  provider: string;
  model: string;
  inputType: string;
  outputTokens: number;
  ttftMs: number | null;
  totalMs: number;
  tokensPerSecAfterFirst: number | null;
  runNumber: number;
  error?: string;
}

const resultsPath = path.join(import.meta.dirname, "..", "results.jsonl");
const lines = fs.readFileSync(resultsPath, "utf-8").trim().split("\n");
const results: TimingResult[] = lines.map(line => JSON.parse(line));

// Filter out errors and group by model
const validResults = results.filter(r => !r.error && r.ttftMs !== null);

interface ModelStats {
  model: string;
  provider: string;
  runs: number;
  avgTtftMs: number;
  minTtftMs: number;
  maxTtftMs: number;
  avgTokPerSec: number;
  avgTotalMs: number;
}

const modelGroups = new Map<string, TimingResult[]>();
for (const r of validResults) {
  const key = `${r.provider}|${r.model}`;
  if (!modelGroups.has(key)) {
    modelGroups.set(key, []);
  }
  modelGroups.get(key)!.push(r);
}

const stats: ModelStats[] = [];
for (const [key, runs] of modelGroups) {
  const [provider, model] = key.split("|");
  const ttfts = runs.map(r => r.ttftMs!);
  const tokSpeeds = runs.filter(r => r.tokensPerSecAfterFirst).map(r => r.tokensPerSecAfterFirst!);
  const totals = runs.map(r => r.totalMs);

  stats.push({
    model,
    provider,
    runs: runs.length,
    avgTtftMs: ttfts.reduce((a, b) => a + b, 0) / ttfts.length,
    minTtftMs: Math.min(...ttfts),
    maxTtftMs: Math.max(...ttfts),
    avgTokPerSec: tokSpeeds.length > 0 ? tokSpeeds.reduce((a, b) => a + b, 0) / tokSpeeds.length : 0,
    avgTotalMs: totals.reduce((a, b) => a + b, 0) / totals.length,
  });
}

// Sort by avg TTFT
stats.sort((a, b) => a.avgTtftMs - b.avgTtftMs);

console.log("═".repeat(120));
console.log("LLM API Latency Summary (Direct API Calls)");
console.log("═".repeat(120));
console.log("");

console.log("┌" + "─".repeat(40) + "┬" + "─".repeat(10) + "┬" + "─".repeat(25) + "┬" + "─".repeat(15) + "┬" + "─".repeat(15) + "┐");
console.log("│ " + "Model".padEnd(38) + " │ " + "Runs".padEnd(8) + " │ " + "TTFT (avg/min/max)".padEnd(23) + " │ " + "Speed (t/s)".padEnd(13) + " │ " + "Total (avg)".padEnd(13) + " │");
console.log("├" + "─".repeat(40) + "┼" + "─".repeat(10) + "┼" + "─".repeat(25) + "┼" + "─".repeat(15) + "┼" + "─".repeat(15) + "┤");

for (const s of stats) {
  const modelName = `${s.provider}/${s.model}`.slice(0, 38);
  const ttftStr = `${(s.avgTtftMs/1000).toFixed(1)}s / ${(s.minTtftMs/1000).toFixed(1)}s / ${(s.maxTtftMs/1000).toFixed(1)}s`;
  const speedStr = `${s.avgTokPerSec.toFixed(0)} t/s`;
  const totalStr = `${(s.avgTotalMs/1000).toFixed(1)}s`;

  console.log("│ " + modelName.padEnd(38) + " │ " + s.runs.toString().padEnd(8) + " │ " + ttftStr.padEnd(23) + " │ " + speedStr.padEnd(13) + " │ " + totalStr.padEnd(13) + " │");
}

console.log("└" + "─".repeat(40) + "┴" + "─".repeat(10) + "┴" + "─".repeat(25) + "┴" + "─".repeat(15) + "┴" + "─".repeat(15) + "┘");

console.log("");
console.log("Notes:");
console.log("- TTFT = Time to First Token");
console.log("- Speed = tokens per second after first token");
console.log("- Lower TTFT is better, higher speed is better");
console.log("- Sonnet 4.5 refused to analyze 'random' input (gibberish) - excluded from those averages");
