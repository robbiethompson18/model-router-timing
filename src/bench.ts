import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import * as fs from "fs";
import * as path from "path";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface TimingResult {
  provider: string;
  model: string;
  inputType: string;
  inputChars: number;
  outputTokens: number;
  requestSentAt: number;
  firstTokenAt: number | null;
  lastTokenAt: number;
  ttftMs: number | null; // time to first token
  totalMs: number;
  tokensPerSecAfterFirst: number | null;
  runNumber: number;
  error?: string;
  timestamp: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

const RUNS_PER_MODEL = 5;
const INPUT_LENGTH = 10000;

// Models to test
const ANTHROPIC_MODELS = [
  "claude-opus-4-6",
  "claude-sonnet-4-5-20250929",
  "claude-haiku-4-5-20251001",
];

const OPENAI_MODELS = [
  "gpt-5.4",
  "gpt-5.4-mini",
];

const GEMINI_MODELS = [
  // "gemini-3.1-pro-preview", // Have 10 runs already; currently timing out at Google (120s+)
  "gemini-3-flash-preview",
];

// OpenRouter-only models (no direct API)
const OPENROUTER_ONLY_MODELS = [
  "moonshotai/kimi-k2.5",
  "minimax/minimax-m2.7",
  "z-ai/glm-4.5",
  "z-ai/glm-4.5-air:free",
];

// Same models via OpenRouter for comparison
const OPENROUTER_COMPARISON_MODELS = [
  "anthropic/claude-opus-4.6",
  "anthropic/claude-sonnet-4.5",
  "anthropic/claude-haiku-4.5",
  "openai/gpt-5.4",
  "openai/gpt-5.4-mini",
  "google/gemini-3.1-pro-preview",
  "google/gemini-3-flash-preview",
];

// ─────────────────────────────────────────────────────────────────────────────
// Input Generators
// ─────────────────────────────────────────────────────────────────────────────

// First 10k characters of Genesis (KJV)
const BIBLE_TEXT = `In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light. And God saw the light, that it was good: and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day. And God said, Let there be a firmament in the midst of the waters, and let it divide the waters from the waters. And God made the firmament, and divided the waters which were under the firmament from the waters which were above the firmament: and it was so. And God called the firmament Heaven. And the evening and the morning were the second day. And God said, Let the waters under the heaven be gathered together unto one place, and let the dry land appear: and it was so. And God called the dry land Earth; and the gathering together of the waters called he Seas: and God saw that it was good. And God said, Let the earth bring forth grass, the herb yielding seed, and the fruit tree yielding fruit after his kind, whose seed is in itself, upon the earth: and it was so. And the earth brought forth grass, and herb yielding seed after his kind, and the tree yielding fruit, whose seed was in itself, after his kind: and God saw that it was good. And the evening and the morning were the third day. And God said, Let there be lights in the firmament of the heaven to divide the day from the night; and let them be for signs, and for seasons, and for days, and years: And let them be for lights in the firmament of the heaven to give light upon the earth: and it was so. And God made two great lights; the greater light to rule the day, and the lesser light to rule the night: he made the stars also. And God set them in the firmament of the heaven to give light upon the earth, And to rule over the day and over the night, and to divide the light from the darkness: and God saw that it was good. And the evening and the morning were the fourth day. And God said, Let the waters bring forth abundantly the moving creature that hath life, and fowl that may fly above the earth in the open firmament of heaven. And God created great whales, and every living creature that moveth, which the waters brought forth abundantly, after their kind, and every winged fowl after his kind: and God saw that it was good. And God blessed them, saying, Be fruitful, and multiply, and fill the waters in the seas, and let fowl multiply in the earth. And the evening and the morning were the fifth day. And God said, Let the earth bring forth the living creature after his kind, cattle, and creeping thing, and beast of the earth after his kind: and it was so. And God made the beast of the earth after his kind, and cattle after their kind, and every thing that creepeth upon the earth after his kind: and God saw that it was good. And God said, Let us make man in our image, after our likeness: and let them have dominion over the fish of the sea, and over the fowl of the air, and over the cattle, and over all the earth, and over every creeping thing that creepeth upon the earth. So God created man in his own image, in the image of God created he him; male and female created he them. And God blessed them, and God said unto them, Be fruitful, and multiply, and replenish the earth, and subdue it: and have dominion over the fish of the sea, and over the fowl of the air, and over every living thing that moveth upon the earth. And God said, Behold, I have given you every herb bearing seed, which is upon the face of all the earth, and every tree, in the which is the fruit of a tree yielding seed; to you it shall be for meat. And to every beast of the earth, and to every fowl of the air, and to every thing that creepeth upon the earth, wherein there is life, I have given every green herb for meat: and it was so. And God saw every thing that he had made, and, behold, it was very good. And the evening and the morning were the sixth day. Thus the heavens and the earth were finished, and all the host of them. And on the seventh day God ended his work which he had made; and he rested on the seventh day from all his work which he had made. And God blessed the seventh day, and sanctified it: because that in it he had rested from all his work which God created and made. These are the generations of the heavens and of the earth when they were created, in the day that the LORD God made the earth and the heavens, And every plant of the field before it was in the earth, and every herb of the field before it grew: for the LORD God had not caused it to rain upon the earth, and there was not a man to till the ground. But there went up a mist from the earth, and watered the whole face of the ground. And the LORD God formed man of the dust of the ground, and breathed into his nostrils the breath of life; and man became a living soul. And the LORD God planted a garden eastward in Eden; and there he put the man whom he had formed. And out of the ground made the LORD God to grow every tree that is pleasant to the sight, and good for food; the tree of life also in the midst of the garden, and the tree of knowledge of good and evil. And a river went out of Eden to water the garden; and from thence it was parted, and became into four heads. The name of the first is Pison: that is it which compasseth the whole land of Havilah, where there is gold; And the gold of that land is good: there is bdellium and the onyx stone. And the name of the second river is Gihon: the same is it that compasseth the whole land of Ethiopia. And the name of the third river is Hiddekel: that is it which goeth toward the east of Assyria. And the fourth river is Euphrates. And the LORD God took the man, and put him into the garden of Eden to dress it and to keep it. And the LORD God commanded the man, saying, Of every tree of the garden thou mayest freely eat: But of the tree of the knowledge of good and evil, thou shalt not eat of it: for in the day that thou eatest thereof thou shalt surely die. And the LORD God said, It is not good that the man should be alone; I will make him an help meet for him. And out of the ground the LORD God formed every beast of the field, and every fowl of the air; and brought them unto Adam to see what he would call them: and whatsoever Adam called every living creature, that was the name thereof. And Adam gave names to all cattle, and to the fowl of the air, and to every beast of the field; but for Adam there was not found an help meet for him. And the LORD God caused a deep sleep to fall upon Adam, and he slept: and he took one of his ribs, and closed up the flesh instead thereof; And the rib, which the LORD God had taken from man, made he a woman, and brought her unto the man. And Adam said, This is now bone of my bones, and flesh of my flesh: she shall be called Woman, because she was taken out of Man. Therefore shall a man leave his father and his mother, and shall cleave unto his wife: and they shall be one flesh. And they were both naked, the man and his wife, and were not ashamed. Now the serpent was more subtil than any beast of the field which the LORD God had made. And he said unto the woman, Yea, hath God said, Ye shall not eat of every tree of the garden? And the woman said unto the serpent, We may eat of the fruit of the trees of the garden: But of the fruit of the tree which is in the midst of the garden, God hath said, Ye shall not eat of it, neither shall ye touch it, lest ye die. And the serpent said unto the woman, Ye shall not surely die: For God doth know that in the day ye eat thereof, then your eyes shall be opened, and ye shall be as gods, knowing good and evil. And when the woman saw that the tree was good for food, and that it was pleasant to the eyes, and a tree to be desired to make one wise, she took of the fruit thereof, and did eat, and gave also unto her husband with her; and he did eat. And the eyes of them both were opened, and they knew that they were naked; and they sewed fig leaves together, and made themselves aprons. And they heard the voice of the LORD God walking in the garden in the cool of the day: and Adam and his wife hid themselves from the presence of the LORD God amongst the trees of the garden. And the LORD God called unto Adam, and said unto him, Where art thou? And he said, I heard thy voice in the garden, and I was afraid, because I was naked; and I hid myself. And he said, Who told thee that thou wast naked? Hast thou eaten of the tree, whereof I commanded thee that thou shouldest not eat? And the man said, The woman whom thou gavest to be with me, she gave me of the tree, and I did eat. And the LORD God said unto the woman, What is this that thou hast done? And the woman said, The serpent beguiled me, and I did eat. And the LORD God said unto the serpent, Because thou hast done this, thou art cursed above all cattle, and above every beast of the field; upon thy belly shalt thou go, and dust shalt thou eat all the days of thy life: And I will put enmity between thee and the woman, and between thy seed and her seed; it shall bruise thy head, and thou shalt bruise his heel. Unto the woman he said, I will greatly multiply thy sorrow and thy conception; in sorrow thou shalt bring forth children; and thy desire shall be to thy husband, and he shall rule over thee. And unto Adam he said, Because thou hast hearkened unto the voice of thy wife, and hast eaten of the tree, of which I commanded thee, saying, Thou shalt not eat of it: cursed is the ground for thy sake; in sorrow shalt thou eat of it all the days of thy life; Thorns also and thistles shall it bring forth to thee; and thou shalt eat the herb of the field; In the sweat of thy face shalt thou eat bread, till thou return unto the ground; for out of it wast thou taken: for dust thou art, and unto dust shalt thou return. And Adam called his wife's name Eve; because she was the mother of all living. Unto Adam also and to his wife did the LORD God make coats of skins, and clothed them.`;

function generateRandomChars(length: number): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>? ";
  let result = "";
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

function generateSlopCode(length: number): string {
  const templates = [
    `function processData${Math.random().toString(36).slice(2)}(input: any) {
  const result = [];
  for (let i = 0; i < input.length; i++) {
    if (input[i] !== null && input[i] !== undefined) {
      result.push(transform(input[i]));
    }
  }
  return result;
}`,
    `class DataHandler${Math.random().toString(36).slice(2)} {
  private cache: Map<string, any> = new Map();

  async fetch(key: string): Promise<any> {
    if (this.cache.has(key)) {
      return this.cache.get(key);
    }
    const data = await this.loadFromSource(key);
    this.cache.set(key, data);
    return data;
  }

  private async loadFromSource(key: string): Promise<any> {
    // TODO: implement actual loading
    return { key, timestamp: Date.now() };
  }
}`,
    `interface Config${Math.random().toString(36).slice(2)} {
  apiUrl: string;
  timeout: number;
  retries: number;
  headers: Record<string, string>;
  enableLogging: boolean;
}

const defaultConfig: Config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
  retries: 3,
  headers: { "Content-Type": "application/json" },
  enableLogging: true,
};`,
    `export async function handleRequest${Math.random().toString(36).slice(2)}(req: Request): Promise<Response> {
  try {
    const body = await req.json();
    const validated = validateInput(body);
    const result = await processBusinessLogic(validated);
    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Request failed:", error);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
    });
  }
}`,
  ];

  let result = "";
  while (result.length < length) {
    result += templates[Math.floor(Math.random() * templates.length)] + "\n\n";
  }
  return result.slice(0, length);
}

// Pad to exactly 10k chars
function padToLength(text: string, targetLength: number): string {
  if (text.length >= targetLength) {
    return text.slice(0, targetLength);
  }
  // Repeat text to reach target
  let result = text;
  while (result.length < targetLength) {
    result += " " + text;
  }
  return result.slice(0, targetLength);
}

const inputs: Record<string, string> = {
  bible: padToLength(BIBLE_TEXT, INPUT_LENGTH),
  random: generateRandomChars(INPUT_LENGTH),
  slop_code: generateSlopCode(INPUT_LENGTH),
};

// ─────────────────────────────────────────────────────────────────────────────
// Prompt that elicits ~1k output tokens
// ─────────────────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a helpful assistant. When given text, analyze it thoroughly and write a detailed essay about its themes, structure, and significance. Your response should be approximately 1000 words (roughly 1200-1500 tokens). Be thorough and detailed.`;

function getUserPrompt(inputText: string): string {
  return `Please analyze the following text in detail. Write approximately 1000 words discussing its themes, structure, patterns, and significance. Be thorough.

---
${inputText}
---

Begin your detailed analysis:`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider Implementations
// ─────────────────────────────────────────────────────────────────────────────

async function benchAnthropic(model: string, inputType: string, inputText: string, runNumber: number): Promise<TimingResult> {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  const requestSentAt = performance.now();
  let firstTokenAt: number | null = null;
  let outputTokens = 0;

  try {
    const stream = await client.messages.stream({
      model,
      max_tokens: 2000,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: getUserPrompt(inputText) }],
    });

    for await (const event of stream) {
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
      }
    }

    const lastTokenAt = performance.now();
    const finalMessage = await stream.finalMessage();
    outputTokens = finalMessage.usage.output_tokens;

    const ttftMs = firstTokenAt ? firstTokenAt - requestSentAt : null;
    const totalMs = lastTokenAt - requestSentAt;
    const timeAfterFirst = firstTokenAt ? lastTokenAt - firstTokenAt : null;
    const tokensPerSecAfterFirst = timeAfterFirst && timeAfterFirst > 0
      ? (outputTokens / (timeAfterFirst / 1000))
      : null;

    return {
      provider: "anthropic",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens,
      requestSentAt,
      firstTokenAt,
      lastTokenAt,
      ttftMs,
      totalMs,
      tokensPerSecAfterFirst,
      runNumber,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    return {
      provider: "anthropic",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens: 0,
      requestSentAt,
      firstTokenAt: null,
      lastTokenAt: performance.now(),
      ttftMs: null,
      totalMs: performance.now() - requestSentAt,
      tokensPerSecAfterFirst: null,
      runNumber,
      error: String(error),
      timestamp: new Date().toISOString(),
    };
  }
}

async function benchOpenAI(model: string, inputType: string, inputText: string, runNumber: number): Promise<TimingResult> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const requestSentAt = performance.now();
  let firstTokenAt: number | null = null;
  let outputTokens = 0;
  let outputText = "";

  try {
    const stream = await client.chat.completions.create({
      model,
      max_completion_tokens: 2000, // GPT-5.x uses max_completion_tokens, not max_tokens
      stream: true,
      stream_options: { include_usage: true },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: getUserPrompt(inputText) },
      ],
    });

    for await (const chunk of stream) {
      if (chunk.choices[0]?.delta?.content) {
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        outputText += chunk.choices[0].delta.content;
      }
      if (chunk.usage) {
        outputTokens = chunk.usage.completion_tokens;
      }
    }

    const lastTokenAt = performance.now();

    // Estimate tokens if not provided
    if (outputTokens === 0) {
      outputTokens = Math.ceil(outputText.length / 4);
    }

    const ttftMs = firstTokenAt ? firstTokenAt - requestSentAt : null;
    const totalMs = lastTokenAt - requestSentAt;
    const timeAfterFirst = firstTokenAt ? lastTokenAt - firstTokenAt : null;
    const tokensPerSecAfterFirst = timeAfterFirst && timeAfterFirst > 0
      ? (outputTokens / (timeAfterFirst / 1000))
      : null;

    return {
      provider: "openai",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens,
      requestSentAt,
      firstTokenAt,
      lastTokenAt,
      ttftMs,
      totalMs,
      tokensPerSecAfterFirst,
      runNumber,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    return {
      provider: "openai",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens: 0,
      requestSentAt,
      firstTokenAt: null,
      lastTokenAt: performance.now(),
      ttftMs: null,
      totalMs: performance.now() - requestSentAt,
      tokensPerSecAfterFirst: null,
      runNumber,
      error: String(error),
      timestamp: new Date().toISOString(),
    };
  }
}

const PER_CALL_TIMEOUT_MS = 180_000; // 3 min max per API call (Gemini thinking can take a while)

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Timeout after ${ms / 1000}s: ${label}`)), ms);
    promise.then(resolve, reject).finally(() => clearTimeout(timer));
  });
}

async function benchGemini(model: string, inputType: string, inputText: string, runNumber: number): Promise<TimingResult> {
  const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY!);

  const requestSentAt = performance.now();
  let firstTokenAt: number | null = null;
  let outputText = "";

  try {
    const genModel = genAI.getGenerativeModel({
      model,
      systemInstruction: SYSTEM_PROMPT,
    });

    const innerRun = async () => {
      const result = await genModel.generateContentStream(getUserPrompt(inputText));

      for await (const chunk of result.stream) {
        const text = chunk.text();
        if (text && firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        outputText += text;
      }

      const lastTokenAt = performance.now();
      const response = await result.response;
      const outputTokens = response.usageMetadata?.candidatesTokenCount ?? Math.ceil(outputText.length / 4);
      return { lastTokenAt, outputTokens };
    };

    const { lastTokenAt, outputTokens } = await withTimeout(innerRun(), PER_CALL_TIMEOUT_MS, `${model}/${inputType}`);

    const ttftMs = firstTokenAt ? firstTokenAt - requestSentAt : null;
    const totalMs = lastTokenAt - requestSentAt;
    const timeAfterFirst = firstTokenAt ? lastTokenAt - firstTokenAt : null;
    const tokensPerSecAfterFirst = timeAfterFirst && timeAfterFirst > 0
      ? (outputTokens / (timeAfterFirst / 1000))
      : null;

    return {
      provider: "gemini",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens,
      requestSentAt,
      firstTokenAt,
      lastTokenAt,
      ttftMs,
      totalMs,
      tokensPerSecAfterFirst,
      runNumber,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    return {
      provider: "gemini",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens: 0,
      requestSentAt,
      firstTokenAt: null,
      lastTokenAt: performance.now(),
      ttftMs: null,
      totalMs: performance.now() - requestSentAt,
      tokensPerSecAfterFirst: null,
      runNumber,
      error: String(error),
      timestamp: new Date().toISOString(),
    };
  }
}

async function benchOpenRouter(model: string, inputType: string, inputText: string, runNumber: number): Promise<TimingResult> {
  // Use BYOK key, not provisioning key (which is for creating keys, not API calls)
  const apiKey = process.env.ROBBIE_DEV_OPENROUTER_KEY_FOR_BYOK || process.env.OPENROUTER_API_KEY;

  const requestSentAt = performance.now();
  let firstTokenAt: number | null = null;
  let outputText = "";

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://usebits.com",
      },
      body: JSON.stringify({
        model,
        max_tokens: 2000,
        stream: true,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: getUserPrompt(inputText) },
        ],
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenRouter error: ${response.status} ${await response.text()}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n").filter(line => line.startsWith("data: "));

      for (const line of lines) {
        const data = line.slice(6);
        if (data === "[DONE]") continue;

        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) {
            if (firstTokenAt === null) {
              firstTokenAt = performance.now();
            }
            outputText += content;
          }
        } catch {
          // Skip parse errors
        }
      }
    }

    const lastTokenAt = performance.now();
    const outputTokens = Math.ceil(outputText.length / 4); // Estimate

    const ttftMs = firstTokenAt ? firstTokenAt - requestSentAt : null;
    const totalMs = lastTokenAt - requestSentAt;
    const timeAfterFirst = firstTokenAt ? lastTokenAt - firstTokenAt : null;
    const tokensPerSecAfterFirst = timeAfterFirst && timeAfterFirst > 0
      ? (outputTokens / (timeAfterFirst / 1000))
      : null;

    return {
      provider: "openrouter",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens,
      requestSentAt,
      firstTokenAt,
      lastTokenAt,
      ttftMs,
      totalMs,
      tokensPerSecAfterFirst,
      runNumber,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    return {
      provider: "openrouter",
      model,
      inputType,
      inputChars: inputText.length,
      outputTokens: 0,
      requestSentAt,
      firstTokenAt: null,
      lastTokenAt: performance.now(),
      ttftMs: null,
      totalMs: performance.now() - requestSentAt,
      tokensPerSecAfterFirst: null,
      runNumber,
      error: String(error),
      timestamp: new Date().toISOString(),
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

function appendResult(result: TimingResult) {
  const outputPath = path.join(import.meta.dirname, "..", "results.jsonl");
  fs.appendFileSync(outputPath, JSON.stringify(result) + "\n");
}

function logResult(result: TimingResult) {
  const status = result.error ? "❌" : "✓";
  const modelShort = result.model.length > 28 ? result.model.slice(0, 25) + "..." : result.model;
  console.log(
    `${status} Run ${result.runNumber}/${RUNS_PER_MODEL} | ${result.provider.padEnd(10)} | ${modelShort.padEnd(28)} | ${result.inputType.padEnd(10)} | ` +
    `TTFT: ${result.ttftMs?.toFixed(0)?.padStart(6) ?? "N/A".padStart(6)}ms | ` +
    `Total: ${(result.totalMs / 1000).toFixed(1).padStart(5)}s | ` +
    `${result.outputTokens.toString().padStart(5)} tok | ` +
    `${result.tokensPerSecAfterFirst?.toFixed(0)?.padStart(4) ?? "N/A".padStart(4)} t/s`
  );
  if (result.error) {
    console.log(`   Error: ${result.error.slice(0, 80)}`);
  }
}

async function runBenchmark(
  name: string,
  benchFn: (model: string, inputType: string, inputText: string, runNumber: number) => Promise<TimingResult>,
  models: string[],
  inputTypes: string[],
) {
  console.log(`\n${"─".repeat(100)}`);
  console.log(`${name}:`);
  console.log(`${"─".repeat(100)}`);

  for (const model of models) {
    for (const inputType of inputTypes) {
      const inputText = inputs[inputType];
      for (let run = 1; run <= RUNS_PER_MODEL; run++) {
        const result = await benchFn(model, inputType, inputText, run);
        logResult(result);
        appendResult(result);
      }
    }
  }
}

async function main() {
  // CLI: pass provider names to run only those (e.g., "gemini openrouter")
  const args = process.argv.slice(2).map(a => a.toLowerCase());
  const runAll = args.length === 0;
  const shouldRun = (name: string) => runAll || args.some(a => name.toLowerCase().includes(a));

  console.log("═".repeat(100));
  console.log("LLM API Latency Benchmark");
  console.log("═".repeat(100));
  console.log(`Input length: ${INPUT_LENGTH} chars`);
  console.log(`Input types: ${Object.keys(inputs).join(", ")}`);
  console.log(`Runs per model: ${RUNS_PER_MODEL}`);
  console.log(`Per-call timeout: ${PER_CALL_TIMEOUT_MS / 1000}s`);
  if (!runAll) console.log(`Filtering to: ${args.join(", ")}`);
  console.log("");

  const inputTypes = Object.keys(inputs);

  // Direct API calls
  if (shouldRun("anthropic") && process.env.ANTHROPIC_API_KEY) {
    await runBenchmark("Anthropic Direct", benchAnthropic, ANTHROPIC_MODELS, inputTypes);
  }

  if (shouldRun("openai") && process.env.OPENAI_API_KEY) {
    await runBenchmark("OpenAI Direct", benchOpenAI, OPENAI_MODELS, inputTypes);
  }

  if (shouldRun("gemini") && process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
    await runBenchmark("Gemini Direct", benchGemini, GEMINI_MODELS, inputTypes);
  }

  // OpenRouter-only models (Kimi, MiniMax, GLM)
  if (shouldRun("openrouter") && (process.env.ROBBIE_DEV_OPENROUTER_KEY_FOR_BYOK || process.env.OPENROUTER_API_KEY)) {
    await runBenchmark("OpenRouter-Only Models", benchOpenRouter, OPENROUTER_ONLY_MODELS, inputTypes);

    // Same models via OpenRouter for comparison
    await runBenchmark("OpenRouter (comparison with direct)", benchOpenRouter, OPENROUTER_COMPARISON_MODELS, inputTypes);
  }

  console.log("\n" + "═".repeat(100));
  console.log("Done! Results appended to results.jsonl");
  console.log("═".repeat(100));
}

main().catch(console.error);
