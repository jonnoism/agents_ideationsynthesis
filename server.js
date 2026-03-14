import 'dotenv/config';
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: '5mb' }));
app.use(express.static(join(__dirname, 'public')));

// --- API clients (lazy-initialized) ---

let anthropic, openai, genAI;

function initClients() {
  if (!anthropic && process.env.ANTHROPIC_API_KEY) {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  }
  if (!openai && process.env.OPENAI_API_KEY) {
    openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  if (!genAI && process.env.GEMINI_API_KEY) {
    genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  }
}

// --- Individual agent callers ---

// maxTokens defaults: 8192 for regular rounds, 16384 for final synthesis

async function callClaude(systemPrompt, userMessage, signal, maxTokens = 8192) {
  if (!anthropic) throw new Error('Anthropic API key not configured');
  const resp = await anthropic.messages.create(
    { model: 'claude-opus-4-5', max_tokens: maxTokens, system: systemPrompt, messages: [{ role: 'user', content: userMessage }] },
    { signal }
  );
  const text = resp.content?.[0]?.text;
  if (!text) throw new Error('Claude returned an empty response');
  return text;
}

async function callChatGPT(systemPrompt, userMessage, signal, maxTokens = 8192) {
  if (!openai) throw new Error('OpenAI API key not configured');
  const resp = await openai.chat.completions.create(
    { model: 'gpt-4o', max_tokens: maxTokens, messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userMessage }] },
    { signal }
  );
  const text = resp.choices?.[0]?.message?.content;
  if (!text) throw new Error('ChatGPT returned an empty response');
  return text;
}

async function callGemini(systemPrompt, userMessage, signal, maxTokens = 8192) {
  if (!genAI) throw new Error('Gemini API key not configured');
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash', systemInstruction: systemPrompt, generationConfig: { maxOutputTokens: maxTokens } });
  const result = await model.generateContent(userMessage, { signal });
  const text = result.response?.text?.();
  if (!text) throw new Error('Gemini returned an empty response');
  return text;
}

const AGENT_CALLERS = [callClaude, callChatGPT, callGemini];
const AGENT_NAMES   = ['claude', 'chatgpt', 'gemini'];

// --- Role rotation system ---
// Minimum 3 rounds so every agent does every role exactly once per cycle.

const ROLES = ['Skeptic', 'Builder', 'Connector'];

// Round is 1-indexed (synthesis rounds). Rotation shifts one seat each round.
function getRole(agentIndex, round) {
  return ROLES[(agentIndex + round - 1) % 3];
}

// Odd synthesis rounds = COMPRESS, even = EXPAND
function isCompressRound(round) {
  return round % 2 === 1;
}

const ROLE_INSTRUCTIONS = {
  Skeptic: `\
YOUR ROLE THIS ROUND: SKEPTIC (Antithesis)
Your primary task is to challenge the previous responses rigorously. Hunt for:
- Flaws in reasoning, logical leaps, or circular arguments
- Unstated assumptions that may not hold under scrutiny
- Edge cases and counterexamples that break or limit the conclusions
- Claims that are overconfident, vague, or under-supported

After identifying what doesn't hold, build your response from what survives scrutiny. You are the antithesis — make every conclusion earn its place.`,

  Builder: `\
YOUR ROLE THIS ROUND: BUILDER (Extension)
Your primary task is to extend. Identify the single strongest idea across all previous responses and develop it significantly further:
- Add depth, specificity, and nuance it currently lacks
- Explore second-order effects and downstream implications
- Supply concrete mechanisms, not just high-level assertions
- Push into territory no previous response has entered

Do not summarize — build on the strongest foundation and add new structure above it.`,

  Connector: `\
YOUR ROLE THIS ROUND: CONNECTOR (Bridge & Emergence)
Your primary task is to bridge. Look for:
- Non-obvious connections between ideas that appeared unrelated
- Apparent contradictions that can both be true simultaneously at different levels
- The unifying pattern beneath what others treated as separate threads
- Whether convergence across agents is genuine insight or lazy consensus

Your most important job is detecting EMERGENCE: the idea that only becomes visible when you combine the previous responses — something more than the sum of parts.`,
};

const UNIVERSAL_CONSTRAINTS = (round) => {
  const passType = isCompressRound(round)
    ? `COMPRESSION PASS (Round ${round} is odd): After your main response, add a "CORE INSIGHT:" section — express the single irreducible idea in 2–3 sentences maximum. Cut everything that is not load-bearing.`
    : `EXPANSION PASS (Round ${round} is even): After your main response, add an "APPLICATIONS:" section — give exactly 3 concrete, specific real-world examples of how the core insight applies in practice. No generalities.`;

  return `\
UNIVERSAL CONSTRAINTS — apply regardless of role:

1. DISPUTE: Before synthesizing, explicitly call out at least one substantive point where you genuinely disagree with the previous round's responses. Label it "DISPUTE:". This must be a real intellectual objection, not a minor quibble.

2. CONFIDENCE LABELS: Tag every key claim with [HIGH], [MED], or [LOW]. For any MED or LOW rating, add a one-phrase reason (e.g. [MED – limited evidence], [LOW – untested assumption]).

3. EMERGENCE: Close your response with an "EMERGENCE:" section naming one genuinely new idea that only appears from combining the previous responses — something not fully present in any single prior answer.

4. ${passType}`;
};

function buildSynthesisPrompt(baseInstructions, round, agentIndex, previousResponses, originalPrompt) {
  const role = getRole(agentIndex, round);

  const collated = AGENT_NAMES
    .map(name => `=== ${name.toUpperCase()} ===\n${previousResponses[name]}`)
    .join('\n\n');

  const systemPrompt = [
    `You are participating in round ${round} of a structured multi-agent synthesis.`,
    `Three agents — Claude, ChatGPT, and Gemini — are collaboratively refining the best possible answer through role-differentiated critique and synthesis.`,
    baseInstructions ? `\nUser focus: ${baseInstructions}\n` : '',
    ROLE_INSTRUCTIONS[role],
    '',
    UNIVERSAL_CONSTRAINTS(round),
  ].join('\n');

  const userMsg = [
    `ORIGINAL PROMPT:\n${originalPrompt}`,
    `\nPREVIOUS ROUND (Round ${round - 1}) RESPONSES:\n${collated}`,
    `\nRespond now according to your role (${role}) and all universal constraints.`,
  ].join('\n');

  return { systemPrompt, userMsg, role };
}

function buildFinalPrompt(baseInstructions, totalRounds, previousResponses, originalPrompt) {
  const collated = AGENT_NAMES
    .map(name => `=== ${name.toUpperCase()} ===\n${previousResponses[name]}`)
    .join('\n\n');

  const systemPrompt = [
    `You are the final synthesizer in a ${totalRounds}-round multi-agent collaborative process.`,
    `Each round used structured role rotation (Skeptic → Builder → Connector) with forced disagreement, confidence labeling, emergence detection, and alternating compression/expansion passes.`,
    baseInstructions ? `\nUser focus: ${baseInstructions}\n` : '',
    `Your task: produce the single definitive answer. This is not another synthesis round — it is the conclusion. Requirements:`,
    `- Distill the highest-confidence insights that survived multi-round scrutiny`,
    `- Resolve any remaining contradictions with explicit reasoning`,
    `- Incorporate the strongest EMERGENCE and CORE INSIGHT findings from across all rounds`,
    `- Be precise, well-structured, and complete — this is the final word`,
    `- Do not hedge unless genuine uncertainty remains after ${totalRounds} rounds of challenge`,
  ].join('\n');

  const userMsg = [
    `ORIGINAL PROMPT:\n${originalPrompt}`,
    `\nFINAL ROUND RESPONSES (after ${totalRounds} synthesis rounds):\n${collated}`,
    `\nProduce the definitive answer.`,
  ].join('\n');

  return { systemPrompt, userMsg };
}

// --- SSE streaming endpoint ---

app.post('/api/synthesize', async (req, res) => {
  initClients();

  let { prompt, rounds = 3, instructions = '' } = req.body;
  rounds = Math.min(20, Math.max(3, parseInt(rounds) || 3));

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const controller = new AbortController();
  const { signal } = controller;

  // Client disconnected (Stop button / browser close) → abort all in-flight calls
  // Use res.on('close') not req.on('close'): req fires as soon as the request body
  // is consumed by express.json(), which would abort the controller immediately.
  res.on('close', () => controller.abort());

  const send = (event, data) => {
    if (!res.writableEnded) res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const isAbort = (err) => err.name === 'AbortError' || signal.aborted;

  try {
    let previousResponses = null;

    for (let round = 0; round <= rounds; round++) {
      if (signal.aborted) break;

      const isFinalRound = round === rounds;
      send('round-start', { round, isFinal: isFinalRound, totalRounds: rounds });

      if (isFinalRound) {
        const { systemPrompt, userMsg } = buildFinalPrompt(instructions, rounds, previousResponses, prompt);
        send('agent-start', { round, agent: 'claude', role: 'Synthesizer', isFinal: true });
        try {
          const result = await callClaude(systemPrompt, userMsg, signal, 16384);
          send('agent-done', { round, agent: 'claude', role: 'Synthesizer', result, isFinal: true });
        } catch (err) {
          if (!isAbort(err)) send('agent-error', { round, agent: 'claude', role: 'Synthesizer', error: err.message, isFinal: true });
        }

      } else if (round === 0) {
        const systemPrompt = [
          `You are a helpful, knowledgeable assistant.`,
          instructions ? `Focus: ${instructions}` : '',
          `Provide a thorough, well-reasoned response. This is the ideation round — be comprehensive and creative.`,
        ].filter(Boolean).join('\n');

        const results = {};
        await Promise.all(AGENT_NAMES.map(async (name, i) => {
          if (signal.aborted) return;
          send('agent-start', { round, agent: name, role: 'Ideation' });
          try {
            const result = await AGENT_CALLERS[i](systemPrompt, prompt, signal);
            results[name] = result;
            send('agent-done', { round, agent: name, role: 'Ideation', result });
          } catch (err) {
            results[name] = `[Error: ${err.message}]`;
            if (!isAbort(err)) send('agent-error', { round, agent: name, role: 'Ideation', error: err.message });
          }
        }));
        previousResponses = results;

      } else {
        const results = {};
        await Promise.all(AGENT_NAMES.map(async (name, i) => {
          if (signal.aborted) return;
          const { systemPrompt, userMsg, role } = buildSynthesisPrompt(instructions, round, i, previousResponses, prompt);
          send('agent-start', { round, agent: name, role });
          try {
            const result = await AGENT_CALLERS[i](systemPrompt, userMsg, signal);
            results[name] = result;
            send('agent-done', { round, agent: name, role, result });
          } catch (err) {
            results[name] = `[Error: ${err.message}]`;
            if (!isAbort(err)) send('agent-error', { round, agent: name, role, error: err.message });
          }
        }));
        previousResponses = results;
      }

      if (!signal.aborted) send('round-done', { round, isFinal: isFinalRound });
    }

    if (signal.aborted) {
      send('terminated', {});
    } else {
      send('done', {});
    }
  } catch (err) {
    if (!isAbort(err)) send('error', { error: err.message });
  }

  res.end();
});

app.get('/api/health', (req, res) => {
  initClients();
  res.json({ claude: !!anthropic, chatgpt: !!openai, gemini: !!genAI });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Multi-Agent Synthesis → http://localhost:${PORT}`));
