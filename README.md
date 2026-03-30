# Perspicacité v2 - AI Research Assistant

**Perspicacité** (French for "insight") is an AI-powered research assistant designed for scientists, researchers, and students who want to find and understand academic literature without getting lost in hundreds of papers.

---

## What is Perspicacité?

Perspicacité helps you:
- 🔍 **Search across academic databases** (PubMed, arXiv, OpenAlex, and more)
- 📚 **Build personal knowledge bases** from papers you care about
- 🤖 **Get AI-powered answers** grounded in real research papers
- 💾 **Save and organize papers** for future reference
- 📖 **Download and read full-text papers** when available

---

## Quick Start (5 minutes)

### Step 1: Get the Software

You need **Python 3.12** installed on your computer. Then:

```bash
# Download the project
git clone <repository-url>
cd perspicacite_v2

# Install dependencies
uv sync --dev

# Or if you don't have uv:
pip install -e ".[dev]"
```

### Step 2: Set Up Your API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add at least one of these:

| Service | Why you need it | Get a key at |
|---------|----------------|--------------|
| **DeepSeek** | AI model for answering questions | [DeepSeek](https://platform.deepseek.com/) |
| **OpenAI** | Alternative AI model | [OpenAI](https://platform.openai.com/) |
| **Anthropic** | Alternative AI model | [Anthropic](https://console.anthropic.com/) |

> 💡 **Don't have API keys?** Start with DeepSeek - they offer free credits for new users.

### Step 3: Start the Web App

```bash
python web_app_full.py
```

Then open your browser and go to: **http://localhost:8000**

---

## How to Use Perspicacité

### 🌐 Using the Web Interface

#### 1. Choose Your Knowledge Base (or Don't!)

In the left sidebar, you'll see a "Knowledge Base" section:

- **"No KB (web search only)"** - Searches the entire web for papers
- **Your own KBs** - Searches only papers you've added

**To create a new Knowledge Base:**
1. Click "+ Create new KB"
2. Choose "Simple" to create an empty KB
3. Or choose "From BibTeX" to import papers from a BibTeX file

#### 2. Ask a Question

Type your research question in the chat box. Examples:
- "What are the effects of green tea extract on metabolism?"
- "How is feature-based molecular networking used in metabolomics?"
- "Compare transformer models to CNNs for medical imaging"

#### 3. Choose a Mode

Select a mode from the dropdown:

| Mode | Best for | Speed |
|------|----------|-------|
| **Basic** | Quick answers from your KB | Fast ⚡ |
| **Advanced** | Better answers with query expansion | Medium |
| **Profound** | Deep research with multiple cycles | Slower |
| **Agentic** | Complex questions needing web search | Variable |

#### 4. Review the Answer

Perspicacité will:
1. Show its "thinking process" (click ▶ to expand)
2. Search relevant papers
3. Filter and score them for relevance
4. Download full texts when possible
5. Generate an answer with citations

#### 5. Save Interesting Papers

At the bottom of each response, you'll see papers found during research. Click "Add to KB" to save them to your knowledge base.

---

## Building Your Knowledge Base

### Method 1: Add Papers from Search Results

When Perspicacité finds papers during research, click the "Add to KB" button on any paper you want to save.

### Method 2: Import from BibTeX

If you have a reference manager (Zotero, Mendeley, EndNote):

1. Export your references as BibTeX (.bib file)
2. In Perspicacité, click "+ Create new KB"
3. Select "From BibTeX" tab
4. Paste your BibTeX content
5. Click "Create from BibTeX"

### Method 3: Add Papers One by One

Coming soon: Upload PDFs directly through the web interface.

For now, use the command line:

```bash
# Add a single PDF to a KB
perspicacite add-pdf <kb-name> <path-to-pdf.pdf>

# Add all PDFs in a folder
perspicacite add-pdf <kb-name> <folder-path>/
```

---

## Understanding the Interface

### Chat History
Your conversations are saved automatically. You can:
- Click any previous chat to resume it
- Click 🗑️ next to "Chat History" to clear all history
- Start a "New Chat" anytime with the button

### Knowledge Base Info
Hover over your selected KB to see:
- Number of papers stored
- Description
- Creation date

### Thinking Messages
When Perspicacité is working, it shows:
- What it's doing (searching, filtering, downloading)
- Progress on paper downloads
- Which sources it's using

---

## Tips for Best Results

### Writing Good Questions

✅ **Good questions:**
- "What are the antioxidant properties of green tea catechins?"
- "How does FBMN compare to traditional molecular networking?"
- "What are recent advances in transformer models for medical imaging?"

❌ **Questions to avoid:**
- "Tell me about tea" (too broad)
- "What is the meaning of life?" (not research-related)
- "Write me an essay" (Perspicacité summarizes research, doesn't write original content)

### Managing Your Knowledge Base

- **Keep KBs focused**: Create separate KBs for different projects
- **Add papers gradually**: Start with 10-20 key papers, expand as needed
- **Review relevance**: Perspicacité scores papers - pay attention to high-scoring ones

### When to Use Each Mode

- **Basic**: You have a well-curated KB and want quick answers
- **Advanced**: Your KB might need broader search
- **Profound**: Complex questions needing multiple perspectives
- **Agentic**: Questions requiring web search beyond your KB

---

## Troubleshooting

### "No API key found" error
You need to set up at least one LLM API key in your `.env` file. See Step 2 above.

### "PDF not available" messages
Not all papers are freely accessible. Perspicacité tries:
1. Unpaywall (open access database)
2. arXiv (preprint server)
3. Direct publisher access
4. Alternative sources (if configured)

### Slow responses
- Try "Basic" or "Advanced" mode instead of "Agentic"
- Check your internet connection
- DeepSeek API can be slow during peak times

### Papers not showing in references
Make sure papers have:
- A valid title
- Author information
- Year of publication

### Can't create Knowledge Base
Knowledge base names must:
- Be unique
- Contain only letters, numbers, hyphens, and underscores
- Not be empty

---

## Configuration

### Using a Different AI Model

Edit `config.yml` to change the AI provider:

```yaml
llm:
  default_provider: "openai"  # or "anthropic", "deepseek"
  default_model: "gpt-4"      # or "claude-3", "deepseek-chat"
```

### Setting Up PDF Download

For better PDF access, you can:

1. **Set your email for Unpaywall** (already done in default config):
   ```yaml
   pdf_download:
     unpaywall_email: "your-email@example.com"
   ```

2. **Add publisher API keys** (optional, for institutional access):
   ```yaml
   pdf_download:
     wiley_tdm_token: "your-token"
     elsevier_api_key: "your-key"
   ```

### Changing the Theme

Click the 🌙/☀️ button in the top right to toggle between light and dark mode.

---

## Privacy & Data

- **Your data stays local**: Knowledge bases are stored on your computer
- **API calls only**: Questions are sent to AI providers (DeepSeek, OpenAI, etc.)
- **No tracking**: We don't collect usage data
- **Your papers**: PDFs you add stay in your local database

---

## Getting Help

### Documentation
- `QUICKSTART.md` - Quick command-line examples
- `TRY_IT_GUIDE.md` - Detailed integration guide
- `IMPLEMENTATION_GUIDE.md` - Technical documentation

### Common Issues

**Problem**: App won't start  
**Solution**: Check that port 8000 isn't already in use, or change it in `config.yml`

**Problem**: "Module not found" errors  
**Solution**: Make sure you ran `uv sync --dev` or `pip install -e ".[dev]"`

**Problem**: AI responses are slow  
**Solution**: Try a different mode (Basic is fastest) or check your internet connection

---

## Contributing

Perspicacité is open source! If you find bugs or want to suggest features:

1. Check existing issues first
2. Create a new issue with:
   - What you were trying to do
   - What happened instead
   - Your operating system and Python version

---

## License

MIT License - Free to use, modify, and distribute.

---

## Acknowledgments

Perspicacité v2 builds on:
- **ChromaDB** for vector storage
- **OpenAlex** for academic search
- **DeepSeek/OpenAI/Anthropic** for AI models
- **Unpaywall** for open access papers

---

**Ready to start?** Run `python web_app_full.py` and ask your first question!
