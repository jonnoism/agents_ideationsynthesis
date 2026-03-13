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

// --- API clients (lazy-initialized on first request) ---

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

async function callClaude(systemPrompt, userMessage) {
  if (!anthropic) throw new Error('Anthropic API key not configured');
  const resp = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    system: systemPrompt,
    messages: [{ role: 'user', content: userMessage }],
  });
  return resp.content[0].text;
}

async function callChatGPT(systemPrompt, userMessage) {
  if (!openai) throw new Error('OpenAI API key not configured');
  const resp = await openai.chat.completions.create({
    model: 'gpt-4o',
    max_tokens: 4096,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ],
  });
  return resp.choices[0].message.content;
}

async function callGemini(systemPrompt, userMessage) {
  if (!genAI) throw new Error('Gemini API key not configured');
  const model = genAI.getGenerativeModel({
    model: 'gemini-2.0-flash',
    systemInstruction: systemPrompt,
  });
  const result = await model.generateContent(userMessage);
  return result.response.text();
}

const AGENTS = {
  claude: callClaude,
  chatgpt: callChatGPT,
  gemini: callGemini,
};

// --- SSE endpoint for the full synthesis pipeline ---

app.post('/api/synthesize', async (req, res) => {
  initClients();

  const { prompt, rounds = 2, instructions = '' } = req.body;

  // SSE setup
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const baseSystem = instructions
    ? `Follow these instructions when generating your response:\n${instructions}\n\n`
    : '';

  try {
    let previousResponses = null;

    for (let round = 0; round <= rounds; round++) {
      const isFinalRound = round === rounds;
      send('round-start', { round, isFinal: isFinalRound });

      // Build the user message for this round
      let userMsg;
      if (round === 0) {
        userMsg = prompt;
      } else {
        const collated = Object.entries(previousResponses)
          .map(([name, text]) => `=== ${name.toUpperCase()} ===\n${text}`)
          .join('\n\n');
        userMsg = `Original prompt:\n${prompt}\n\nHere are the ${round === 1 ? 'initial' : 'previous round\'s'} responses from all three agents:\n\n${collated}\n\nSynthesize the best elements from all three responses into a single, improved answer. Keep what's strongest, discard what's weak, and resolve any contradictions.`;
      }

      if (isFinalRound) {
        // Final round: Claude only
        const systemPrompt = `${baseSystem}You are the final synthesizer. You have seen multiple rounds of collaborative refinement between three AI agents (Claude, ChatGPT, Gemini). Produce the definitive, best possible answer. Be thorough, precise, and well-structured.`;
        send('agent-start', { round, agent: 'claude', isFinal: true });
        try {
          const result = await callClaude(systemPrompt, userMsg);
          send('agent-done', { round, agent: 'claude', result, isFinal: true });
        } catch (err) {
          send('agent-error', { round, agent: 'claude', error: err.message, isFinal: true });
        }
      } else {
        // Regular round: all 3 agents in parallel
        const systemPrompt = round === 0
          ? `${baseSystem}You are a helpful, knowledgeable assistant. Provide a thorough, well-reasoned response.`
          : `${baseSystem}You are participating in a multi-agent collaborative synthesis. Review all three previous responses and produce an improved, synthesized answer.`;

        const agentNames = ['claude', 'chatgpt', 'gemini'];
        const results = {};

        // Launch all 3 in parallel
        const promises = agentNames.map(async (name) => {
          send('agent-start', { round, agent: name });
          try {
            const result = await AGENTS[name](systemPrompt, userMsg);
            results[name] = result;
            send('agent-done', { round, agent: name, result });
          } catch (err) {
            results[name] = `[Error: ${err.message}]`;
            send('agent-error', { round, agent: name, error: err.message });
          }
        });

        await Promise.all(promises);
        previousResponses = results;
      }

      send('round-done', { round, isFinal: isFinalRound });
    }

    send('done', {});
  } catch (err) {
    send('error', { error: err.message });
  }

  res.end();
});

// Health check
app.get('/api/health', (req, res) => {
  initClients();
  res.json({
    claude: !!anthropic,
    chatgpt: !!openai,
    gemini: !!genAI,
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Multi-Agent Synthesis running at http://localhost:${PORT}`);
});
