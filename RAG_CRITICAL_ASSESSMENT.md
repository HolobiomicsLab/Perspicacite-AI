# 🔴 CRITICAL: RAG Implementation Assessment & Improvement Plan

**Priority:** P0 (Highest)  
**Status:** Significant Regression from v1  
**Owner:** Future Coding Agent  
**Date:** 2026-03-17  

---

## Executive Summary

The v2 DeepRAG implementation is a **critical regression** from v1 Profound mode. While the architecture is modern (async, protocol-based), the actual agentic RAG capabilities have been stripped down to ~15% of v1 functionality. **This is the core differentiator of the product and must receive immediate attention.**

**Severity: CRITICAL** — Without these improvements, v2 offers no advantage over basic RAG systems.

---

## Current State Analysis

### v1 Profound vs v2 DeepRAG Comparison

| Capability | v1 Profound | v2 DeepRAG | Status | Priority |
|------------|-------------|------------|--------|----------|
| **Document Quality Assessment** | ✅ Full LLM analysis with confidence | ❌ Not implemented | **MISSING** | P0 |
| **Early Exit Logic** | ✅ Confidence-based termination | ❌ Max iterations only | **MISSING** | P0 |
| **Plan Adaptation** | ✅ Dynamic plan adjustment based on findings | ❌ Stub (`return current_plan`) | **MISSING** | P0 |
| **Web Search Fallback** | ✅ Automatic when RAG insufficient | ❌ Not implemented | **MISSING** | P0 |
| **Two-Stage RAG** | ✅ Basic → Advanced progression | ❌ Single stage | **MISSING** | P1 |
| **Iteration Memory** | ✅ Summary with findings/missing info | ❌ Not tracked | **MISSING** | P1 |
| **Unanswerable Detection** | ✅ Detects false premises | ❌ Not implemented | **MISSING** | P1 |
| **Tool Integration** | ✅ Direct tool calls | ⚠️ Tools defined but UNUSED | **BROKEN** | P0 |
| **Structured Planning** | ✅ JSON with reasoning | ❌ Line-based parsing | **DEGRADED** | P1 |

### Code Size Comparison

```
v1 Profound:  ~1,634 lines (mature, battle-tested)
v2 DeepRAG:    ~225 lines  (placeholder implementation)

Coverage: ~14% of v1 capabilities migrated
```

---

## Critical Issues (Must Fix)

### 1. 🔴 Tools Are Defined But Never Used

**File:** `src/perspicacite/rag/modes/deep.py`

```python
# PROBLEM: Tools passed but ignored
async def execute(self, ..., tools: Any, ...):
    # 'tools' parameter is NEVER used!
    # No tools.get("web_search"), no dynamic tool selection
```

**Impact:** Web search, citation networks, PDF fetching are all non-functional.

**Fix:** Implement tool orchestration logic:
```python
# Check if KB search sufficient
kb_results = await tools.get("kb_search").execute(...)
if not sufficient and request.use_web_search:
    web_results = await tools.get("web_search").execute(...)
```

### 2. 🔴 No Document Quality Assessment

**File:** `src/perspicacite/rag/modes/deep.py`

**v1 had:** `profonde.py:410-545` - Full `_analyze_documents()` method

**v2 needs:**
- Port `_analyze_documents()` method
- Add `purpose_fulfilled`, `question_answered`, `confidence` tracking
- Use LLM to evaluate document relevance

### 3. 🔴 No Early Exit Logic

**File:** `src/perspicacite/rag/modes/deep.py:66-67`

**Current (broken):**
```python
if iteration >= max_iterations:
    break  # Only exit condition!
```

**v1 had:** `profonde.py:976-1068` - Full `_is_question_answered()` with LLM evaluation

**Fix:** Add early exit when question is answered with confidence:
```python
question_answered, confidence = await self._is_question_answered(steps, request.query)
if question_answered and confidence >= self.early_exit_confidence:
    break
```

### 4. 🔴 Plan Adaptation is a Stub

**File:** `src/perspicacite/rag/modes/deep.py:160-170`

**Current:**
```python
async def _adjust_plan(self, ...):
    """Adjust research plan based on findings."""
    # For now, return same plan
    # Full implementation would analyze gaps
    return current_plan  # NO-OP!
```

**v1 had:** `profonde.py:1070-1253` - Full `_review_and_adjust_plan()` with:
- Consecutive failure detection
- Unanswerable question detection  
- Plan modification based on findings
- Strategy change tracking

### 5. 🔴 No Web Search Integration

**File:** `src/perspicacite/rag/modes/deep.py`

Despite `use_web_search` parameter in `RAGRequest`, it's never checked.

**v1 had:** Three-stage pipeline (Basic RAG → Advanced RAG → Web Search)

**Fix:** Port `_process_documents()` from v1 with all three stages.

---

## Implementation Roadmap

### Phase 1: Restore Core Agentic Behavior (P0)

**Files to modify:**
1. `src/perspicacite/rag/modes/deep.py` - Main implementation
2. `src/perspicacite/rag/modes/base.py` - Add shared methods if needed

**Tasks:**
1. [ ] Port `_analyze_documents()` from v1 profonde.py:410-545
2. [ ] Port `_is_question_answered()` from v1 profonde.py:976-1068
3. [ ] Port `_review_and_adjust_plan()` from v1 profonde.py:1070-1253
4. [ ] Port `_create_iteration_summary()` from v1 profonde.py:621-707
5. [ ] Implement tool usage in `execute()` method
6. [ ] Add web search fallback logic
7. [ ] Fix `_create_research_plan()` to use structured JSON parsing

### Phase 2: Enhance Architecture (P1)

1. [ ] Add proper async streaming for agentic steps
2. [ ] Implement state persistence between iterations
3. [ ] Add metrics/logging for agentic decisions
4. [ ] Write comprehensive tests for agentic behavior

### Phase 3: Advanced Features (P2)

1. [ ] Response refinement (from v1)
2. [ ] Relevancy optimization (from v1)
3. [ ] Multi-strategy document retrieval

---

## Reference Implementation

**Source of Truth:** `/mnt/d/new_repos/perspicacite_v2/packages_to_use/Perspicacite-AI-release/core/profonde.py`

This file contains the complete, working implementation. The task is to:
1. **Adapt** it to the new async architecture
2. **Integrate** it with the tool system
3. **Modernize** it with proper type hints

**Key methods to port:**
- `_analyze_documents()` - Lines 410-545
- `_create_research_plan()` - Lines 547-619
- `_create_iteration_summary()` - Lines 621-707
- `_is_question_answered()` - Lines 976-1068
- `_review_and_adjust_plan()` - Lines 1070-1253
- `_process_documents()` - Lines 179-329 (three-stage retrieval)

---

## Agent Swarm Feature (Future Enhancement)

**Status:** Planned for v2.2+  
**Skill Reference:** `/home/tjiang/.config/agents/skills/agent-swarm/SKILL.md`

### Vision
Implement true multi-agent research using the agent-swarm skill patterns:

```
User Query
    ↓
[Research Coordinator Agent]
    ↓
    ├── [Literature Search Agent] → SciLEx APIs
    ├── [Document Analysis Agent] → PDF parsing
    ├── [Citation Network Agent] → CrossRef/OpenCitations
    └── [Synthesis Agent] → Answer generation
    ↓
[Integration & Verification]
    ↓
Final Answer
```

### Swarm Patterns to Implement

#### 1. Coordinator-Worker Pattern
```python
# Coordinator analyzes query and delegates
research_tasks = []
for sub_question in decomposed_queries:
    task = Task(
        description=f"Research: {sub_question}",
        prompt=f"Search knowledge base and web for: {sub_question}",
        subagent_name="researcher"
    )
    research_tasks.append(task)

# Aggregate findings
findings = [t.result for t in research_tasks]
answer = synthesize(findings)
```

#### 2. Maker-Checker Pattern (for quality)
```python
# Maker: Generate answer
answer = await generate_answer(query, docs)

# Checker: Verify completeness
verification = await verify_answer(answer, query, docs)

if not verification.approved:
    # Refine and recheck
    answer = await refine_answer(answer, verification.feedback)
```

#### 3. Fan-out/Fan-in (for large KBs)
```python
# Search multiple collections in parallel
search_tasks = []
for kb in knowledge_bases:
    task = Task(
        description=f"Search {kb}",
        prompt=f"Search {kb} for: {query}",
        subagent_name="researcher"
    )
    search_tasks.append(task)

# Merge results
all_results = merge_and_rerank([t.result for t in search_tasks])
```

### Integration Points

| Swarm Component | Perspicacité Integration |
|-----------------|-------------------------|
| Research Coordinator | New `CoordinatorAgent` class in `rag/agents/` |
| Literature Searcher | SciLEx adapter + Web search tools |
| Document Analyst | PDF parser + Chunking pipeline |
| Citation Analyst | CitationNetworkTool |
| Synthesizer | LLM client with context building |
| Verifier | New verification layer |

### Files to Create (Future)
```
src/perspicacite/rag/agents/
├── __init__.py
├── coordinator.py      # Main orchestrator
├── researcher.py       # Literature search agent
├── analyst.py          # Document analysis agent
├── synthesizer.py      # Answer synthesis agent
└── verifier.py         # Quality check agent
```

---

## Testing Requirements

### Unit Tests (P0)
- [ ] `test_deep_rag_analysis.py` - Document quality assessment
- [ ] `test_deep_rag_planning.py` - Research plan creation/adaptation
- [ ] `test_deep_rag_early_exit.py` - Early termination logic
- [ ] `test_deep_rag_tools.py` - Tool orchestration

### Integration Tests (P1)
- [ ] Full research cycle with early exit
- [ ] Web search fallback when KB insufficient
- [ ] Plan adaptation after failed steps
- [ ] Streaming agentic events

### Regression Tests
- [ ] Verify v1 test cases still pass
- [ ] Compare output quality v1 vs v2

---

## Definition of Done

DeepRAG is complete when:
1. ✅ All P0 features from v1 are ported and working
2. ✅ Tools are actually invoked during research
3. ✅ Early exit works based on confidence, not just max iterations
4. ✅ Plan adaptation modifies strategy based on findings
5. ✅ Web search automatically triggers when KB insufficient
6. ✅ Document quality is assessed for each retrieval
7. ✅ All tests pass with >80% coverage
8. ✅ Performance is comparable or better than v1

---

## Notes for Future Agents

1. **This is the CORE FEATURE** — Perspicacité's value proposition is agentic RAG, not basic retrieval.

2. **v1 is the reference** — The code in `packages_to_use/Perspicacite-AI-release/core/profonde.py` works. Study it carefully.

3. **Preserve the architecture** — Keep the async, protocol-based design, but fill in the missing logic.

4. **Test incrementally** — Add one feature at a time and test thoroughly.

5. **Agent swarm is future** — Don't let swarm complexity distract from fixing basic DeepRAG first.

---

**Related Documents:**
- `/mnt/d/new_repos/perspicacite_v2/Perspicacite_v2_Planning.md` - Original design (Section 5: RAG Modes Analysis)
- `/mnt/d/new_repos/perspicacite_v2/IMPLEMENTATION_GUIDE.md` - Implementation spec (Section 5.4: DeepRAG)
- `/home/tjiang/.config/agents/skills/agent-swarm/SKILL.md` - Swarm orchestration patterns
- `/mnt/d/new_repos/perspicacite_v2/packages_to_use/Perspicacite-AI-release/core/profonde.py` - v1 reference implementation
