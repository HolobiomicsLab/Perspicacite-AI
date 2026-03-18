# Dynamic KB & Agentic RAG Workflow Flowcharts

## Flowchart 1: High-Level Dynamic KB Building Workflow

```mermaid
flowchart TD
    Start([User Query]) --> Search[SciLEx Search]
    Search --> Results{Papers<br/>Found?}
    Results -->|No| Refine[Query Refinement]
    Refine --> Search
    Results -->|Yes| Download[PDF Download]
    
    Download --> Assess[Relevance Assessment]
    Assess --> Relevant{Sufficient<br/>Relevant<br/>Papers?}
    
    Relevant -->|No| Refine
    Relevant -->|Yes| KB[Build Dynamic KB]
    
    KB --> Chunk[Chunk & Embed Papers]
    Chunk --> Store[Store in Session<br/>Vector Collection]
    
    Store --> Retrieve[Retrieve from KB]
    Retrieve --> Generate[Generate Answer]
    Generate --> Cleanup[Cleanup Session KB]
    Cleanup --> End([End])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style KB fill:#87CEEB
    style Assess fill:#FFD700
```

## Flowchart 2: Detailed Agentic RAG Workflow

```mermaid
flowchart TD
    Start([User Query]) --> Plan[Create Research Plan]
    Plan --> PlanSteps[Generate 2-4<br/>Research Steps]
    
    subgraph Cycle [Research Cycle]
        direction TB
        
        StepStart[For Each Step] --> ToolSelect{Select Tool}
        
        ToolSelect -->|KB First| KBSearch[KB Vector Search]
        KBSearch --> Quality{Document<br/>Quality Check}
        
        Quality -->|Sufficient| Analyze[Analyze Documents]
        Quality -->|Insufficient| WebSearch[Web Search Fallback]
        WebSearch --> PDFDownload[Download & Parse PDFs]
        PDFDownload --> Analyze
        
        Analyze --> Extract[Extract Key Points]
        Extract --> StepEval{Step<br/>Successful?}
        
        StepEval -->|Yes| MarkSuccess[Mark Success]
        StepEval -->|No| MarkFail[Mark Failed]
        
        MarkSuccess --> EarlyCheck{Question<br/>Answered?}
        MarkFail --> EarlyCheck
        
        EarlyCheck -->|Yes & Confident| ExitCycle[Exit Early]
        EarlyCheck -->|No| NextStep{More<br/>Steps?}
        
        NextStep -->|Yes| StepStart
        NextStep -->|No| ExitCycle
    end
    
    PlanSteps --> Cycle
    
    Cycle --> Summary[Create Iteration Summary]
    Summary --> Continue{Should<br/>Continue?}
    
    Continue -->|Yes & Max Not Reached| ReviewPlan[Review & Adjust Plan]
    ReviewPlan --> Plan
    
    Continue -->|No| FinalAnswer[Generate Final Answer]
    Continue -->|Max Reached| FinalAnswer
    
    FinalAnswer --> End([End])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style KBSearch fill:#87CEEB
    style WebSearch fill:#DDA0DD
    style Analyze fill:#FFD700
    style Quality fill:#FFA07A
    style EarlyCheck fill:#98FB98
```

## Flowchart 3: Document Quality Assessment (Sub-process)

```mermaid
flowchart TD
    Start([Docs Retrieved]) --> Stage1[Stage 1: Basic RAG]
    
    Stage1 --> Assess1{Quality<br/>Sufficient?}
    Assess1 -->|Yes| Return1[Return Documents]
    
    Assess1 -->|No| Stage2[Stage 2: Advanced RAG]
    Stage2 --> Contextual[Generate Contextual<br/>Queries from Missing Aspects]
    Contextual --> SearchAdv[Search with<br/>Contextual Queries]
    
    SearchAdv --> Assess2{Quality<br/>Sufficient?}
    Assess2 -->|Yes| Return2[Return Documents]
    
    Assess2 -->|No| Stage3[Stage 3: Web Search]
    Stage3 --> WebQuery[Query Web Search API]
    WebQuery --> AssessPDF[Assess PDF Quality]
    AssessPDF --> Combine[Combine with RAG Results]
    Combine --> Return3[Return Combined Documents]
    
    Return1 --> End([End])
    Return2 --> End
    Return3 --> End
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Stage1 fill:#87CEEB
    style Stage2 fill:#87CEEB
    style Stage3 fill:#DDA0DD
    style Assess1 fill:#FFD700
    style Assess2 fill:#FFD700
```

## Flowchart 4: Plan Review & Adjustment (Sub-process)

```mermaid
flowchart TD
    Start([Review Triggered]) --> EvalProgress[Evaluate Research Progress]
    
    EvalProgress --> CheckQType{Question Type}
    
    CheckQType -->|Answerable| Continue[Continue Research]
    CheckQType -->|Partially Answerable| Partial[Complete with<br/>Limitations Note]
    CheckQType -->|Unanswerable| Unanswer[Explain<br/>Unanswerable]
    CheckQType -->|False Premise| FalsePremise[Explain<br/>False Premise]
    
    Continue --> CheckPlan{Current Plan<br/>Working?}
    
    CheckPlan -->|Yes| KeepPlan[Keep Original Plan]
    CheckPlan -->|No| AdjustPlan[Adjust Plan Based<br/>on Findings]
    
    KeepPlan --> ReturnPlan[Return Updated Plan]
    AdjustPlan --> ReturnPlan
    Partial --> End([End])
    Unanswer --> End
    FalsePremise --> End
    ReturnPlan --> End
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style CheckQType fill:#FFD700
    style CheckPlan fill:#FFD700
    style Continue fill:#98FB98
```

## Flowchart 5: Complete System Architecture

```mermaid
flowchart TB
    subgraph Input [Input Layer]
        User[User Query]
        Config[Configuration]
    end
    
    subgraph Planning [Planning Layer]
        CreatePlan[Create Research Plan]
        AdjustPlan[Review & Adjust Plan]
    end
    
    subgraph Execution [Execution Layer]
        subgraph Step1 [Research Step]
            ToolSelect[Tool Selection]
            KBSearch[KB Search]
            WebSearch[Web Search]
            PDFParse[PDF Parsing]
        end
        
        Quality[Quality Assessment]
        Analysis[Document Analysis]
    end
    
    subgraph Evaluation [Evaluation Layer]
        EarlyExit{Early Exit<br/>Check}
        IterSummary[Iteration Summary]
        ContinueCheck{Continue?}
    end
    
    subgraph Knowledge [Knowledge Layer]
        DynamicKB[Dynamic KB Builder]
        Chunking[Document Chunking]
        Embedding[Embedding Generation]
        VectorStore[(Session Vector Store)]
    end
    
    subgraph Output [Output Layer]
        AnswerGen[Answer Generation]
        Sources[Source Citations]
        Cleanup[Session Cleanup]
        Response[Final Response]
    end
    
    User --> CreatePlan
    Config --> CreatePlan
    
    CreatePlan --> ToolSelect
    
    ToolSelect -->|Primary| KBSearch
    ToolSelect -->|Fallback| WebSearch
    WebSearch --> PDFParse
    
    KBSearch --> Quality
    PDFParse --> Quality
    
    Quality -->|Pass| Analysis
    Quality -->|Fail| WebSearch
    
    Analysis --> DynamicKB
    
    DynamicKB --> Chunking
    Chunking --> Embedding
    Embedding --> VectorStore
    
    Analysis --> EarlyExit
    EarlyExit -->|Yes| AnswerGen
    EarlyExit -->|No| IterSummary
    
    IterSummary --> ContinueCheck
    ContinueCheck -->|Yes| AdjustPlan
    ContinueCheck -->|No| AnswerGen
    AdjustPlan --> ToolSelect
    
    VectorStore --> AnswerGen
    AnswerGen --> Sources
    Sources --> Cleanup
    Cleanup --> Response
    
    style User fill:#90EE90
    style Response fill:#FFB6C1
    style DynamicKB fill:#87CEEB
    style VectorStore fill:#DDA0DD
    style Quality fill:#FFD700
    style EarlyExit fill:#98FB98
```

## Legend

| Color | Meaning |
|-------|---------|
| 🟢 Green | Start/End points |
| 🔵 Blue | Primary process steps |
| 🟡 Yellow | Decision/Assessment points |
| 🟣 Purple | External services (Web, Vector DB) |
| 🟠 Orange | Sub-process calls |
| 🔴 Pink | Output/Result |

## Key Workflow Features

### 1. **Dynamic KB Building**
- Session-scoped vector collections
- Automatic chunking and embedding
- Cleanup after use

### 2. **Agentic Capabilities**
- **Document Quality Assessment**: 3-stage retrieval (Basic → Advanced → Web)
- **Early Exit**: Stops when question is confidently answered
- **Tool Selection**: Intelligent choice between KB and web search
- **Plan Adjustment**: Dynamic strategy modification based on findings

### 3. **Iterative Research**
- Multiple research cycles
- Plan refinement between cycles
- Progress evaluation at each step

### 4. **Graceful Degradation**
- Web search fallback when KB insufficient
- Query refinement when no results
- Clear explanations for unanswerable questions
