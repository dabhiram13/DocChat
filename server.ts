import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import Groq from 'groq-sdk';
import { pipeline, env } from '@xenova/transformers';

// Configure transformers to not look for local models
env.allowLocalModels = false;

// Simple Vector Store
interface ChunkRecord {
  id: string;
  docId: string;
  text: string;
  embedding: number[];
}
const vectorStore: ChunkRecord[] = [];

// Initialize HF Pipeline once
let extractor: any = null;
async function getExtractor() {
  if (!extractor) {
    console.log("Loading embedding model (this may take a moment on first run)...");
    extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', {
      quantized: true,
    });
    console.log("Embedding model loaded.");
  }
  return extractor;
}

// Ensure the model starts loading in the background
getExtractor().catch(console.error);

function cosineSimilarity(a: number[], b: number[]) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Simple overlap chunker
function chunkText(text: string, maxTokens: number = 250, overlap: number = 50) {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += (maxTokens - overlap)) {
    chunks.push(words.slice(i, i + maxTokens).join(' '));
  }
  return chunks;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  const upload = multer({ storage: multer.memoryStorage() });

  // Get groq client lazily
  function getGroqClient() {
    if (!process.env.GROQ_API_KEY) {
      throw new Error("GROQ_API_KEY is not defined in environment variables.");
    }
    return new Groq({ apiKey: process.env.GROQ_API_KEY });
  }

  // API Route: Upload PDF & Create Embeddings
  app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      console.log(`Processing ${req.file.originalname}`);
      const pdfData = await pdfParse(req.file.buffer);
      const text = pdfData.text.replace(/\0/g, ''); // Clean bad chars

      console.log(`Extracted ${text.length} characters.`);
      const chunks = chunkText(text, 200, 50); // smaller chunks are often better for bge-small
      console.log(`Created ${chunks.length} chunks.`);

      const model = await getExtractor();
      const docId = Math.random().toString(36).substring(7);

      for (let i = 0; i < chunks.length; i++) {
        // Feature extraction output is a Tensor. Need to get the array.
        // For pooling, we can use 'mean'
        const output = await model(chunks[i], { pooling: 'mean', normalize: true });
        const embedding = Array.from(output.data) as number[];
        
        vectorStore.push({
          id: `${docId}-${i}`,
          docId,
          text: chunks[i],
          embedding
        });
      }

      res.json({ message: 'Document processed successfully', docId, chunks: chunks.length });
    } catch (error: any) {
      console.error("Upload error:", error);
      res.status(500).json({ error: error.message || 'Error processing document' });
    }
  });

  // API Route: Chat and Retrieval
  app.post('/api/chat', async (req, res) => {
    try {
      const { message, history = [], docId } = req.body;
      if (!message) {
        return res.status(400).json({ error: 'Message is required' });
      }

      let contextText = '';
      let citations: string[] = [];

      // Only retrieve if we have documents uploaded and a valid docId
      if (docId) {
        const model = await getExtractor();
        
        // Add specific instruction for retrieval if using BGE
        // Wait, bge-small-en-v1.5 expects queries to potentially have a prompt, but let's just use the query directly.
        const output = await model(message, { pooling: 'mean', normalize: true });
        const queryEmbedding = Array.from(output.data) as number[];

        const docChunks = vectorStore.filter(c => c.docId === docId);
        
        const scoredChunks = docChunks.map(chunk => ({
          ...chunk,
          score: cosineSimilarity(queryEmbedding, chunk.embedding)
        })).sort((a, b) => b.score - a.score);

        const topChunks = scoredChunks.slice(0, 4);
        contextText = topChunks.map(c => c.text).join('\n\n');
        citations = topChunks.map(c => c.text);
      }

      const groq = getGroqClient();

      const systemPrompt = `You are a helpful AI assistant that answers questions based on the provided document excerpts.
If the answer is not contained within the given excerpts, say "I don't know based on the provided document." 
Do NOT make up information. Be concise and clear.

Document Excerpts:
${contextText || "No context provided."}`;

      const apiMessages = [
        { role: 'system', content: systemPrompt },
        ...history.map((msg: any) => ({ role: msg.role, content: msg.content })),
        { role: 'user', content: message }
      ];

      const completion = await groq.chat.completions.create({
        messages: apiMessages,
        model: 'llama-3.3-70b-versatile',
        temperature: 0.1,
      });

      const responseText = completion.choices[0]?.message?.content || '';

      res.json({ 
        response: responseText,
        citations: citations
      });

    } catch (error: any) {
      console.error("Chat error:", error);
      res.status(500).json({ error: error.message || 'Error processing chat' });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static('dist'));
    app.get('*', (req, res) => {
      res.sendFile(path.join(process.cwd(), 'dist/index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch(console.error);
