"""Comprehensive tests for all RAG modes.

Tests verify:
- All RAG mode files exist and have correct structure
- Prompts are defined correctly
- Config schema is set up properly
- Engine supports all modes
"""

import os
import sys


def test_rag_modes_files_exist():
    """Test that all RAG mode files exist."""
    modes_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'rag', 'modes')
    
    required_files = [
        'base.py',
        'basic.py',
        'advanced.py', 
        'profound.py',
        'agentic.py',
        '__init__.py'
    ]
    
    for file in required_files:
        filepath = os.path.join(modes_dir, file)
        assert os.path.exists(filepath), f"Missing {file}"
        print(f"  ✅ {file} exists")
        
    print("✅ All RAG mode files exist\n")


def test_mode_class_structure():
    """Test that mode classes have expected structure."""
    modes_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'rag', 'modes')
    
    expected_methods = {
        'basic.py': ['execute', 'execute_stream', '_generate_response'],
        'advanced.py': ['execute', 'execute_stream', '_generate_similar_queries', '_wrrf_retrieval'],
        'profound.py': ['execute', 'execute_stream', '_create_plan', '_execute_step'],
        'agentic.py': ['execute', 'execute_stream', '_create_research_plan', '_execute_step'],
    }
    
    for file, methods in expected_methods.items():
        filepath = os.path.join(modes_dir, file)
        with open(filepath, 'r') as f:
            content = f.read()
        
        print(f"  Checking {file}...")
        for method in methods:
            assert f'def {method}' in content, f"{file} missing {method}"
            print(f"    ✅ {method}")
    
    print("✅ All mode classes have expected structure\n")


def test_prompts_file():
    """Test that prompts.py exists and has required prompts."""
    prompts_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'rag', 'prompts.py')
    assert os.path.exists(prompts_path), "prompts.py not found"
    
    with open(prompts_path, 'r') as f:
        content = f.read()
    
    # Check for key prompts
    required_prompts = [
        'MANDATORY_PROMPT',
        'DEFAULT_SYSTEM_PROMPT', 
        'FORMAT_PROMPT',
        'GENERATE_SIMILAR_QUERIES_PROMPT',
        'EVALUATE_RESPONSE_PROMPT',
        'PROFOUND_CREATE_PLAN_PROMPT',
        'PROFOUND_IS_QUESTION_ANSWERED_PROMPT',
    ]
    
    print("  Checking prompts.py...")
    for prompt in required_prompts:
        assert prompt in content, f"Missing {prompt}"
        print(f"    ✅ {prompt}")
    
    print("✅ Prompts file exists with all required prompts\n")


def test_hybrid_retrieval():
    """Test that hybrid retrieval module exists and has key functions."""
    hybrid_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'retrieval', 'hybrid.py')
    assert os.path.exists(hybrid_path), "hybrid.py not found"
    
    with open(hybrid_path, 'r') as f:
        content = f.read()
    
    # Check for key functions
    required_functions = [
        'def normalize_scores',
        'def combine_scores',
        'def compute_bm25_scores',
        'def hybrid_retrieval',
    ]
    
    print("  Checking hybrid.py...")
    for func in required_functions:
        assert func in content, f"Missing {func}"
        print(f"    ✅ {func}")
    
    print("✅ Hybrid retrieval module exists\n")


def test_config_schema():
    """Test that config schema has RAG mode settings."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'config', 'schema.py')
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check for RAG mode configurations
    print("  Checking config/schema.py...")
    assert 'class RAGModesConfig' in content
    print("    ✅ RAGModesConfig class")
    
    for mode in ['basic:', 'advanced:', 'profound:', 'agentic:']:
        assert mode in content, f"Missing {mode}"
        print(f"    ✅ {mode} setting")
    
    print("✅ Config schema has RAG mode settings\n")


def test_engine_imports_all_modes():
    """Test that RAGEngine supports all modes."""
    engine_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'rag', 'engine.py')
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    print("  Checking engine.py...")
    required_modes = ['BasicRAGMode', 'AdvancedRAGMode', 'ProfoundRAGMode', 'AgenticRAGMode']
    
    for mode in required_modes:
        assert mode in content, f"Missing {mode}"
        print(f"    ✅ {mode}")
    
    # Check that modes are registered
    assert 'RAGMode.BASIC' in content
    assert 'RAGMode.ADVANCED' in content
    assert 'RAGMode.PROFOUND' in content
    assert 'RAGMode.AGENTIC' in content
    print("    ✅ All RAGMode enum values registered")
    
    print("✅ RAGEngine supports all modes\n")


def test_rag_model_enum():
    """Test RAGMode enum values."""
    rag_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'models', 'rag.py')
    
    with open(rag_path, 'r') as f:
        content = f.read()
    
    print("  Checking models/rag.py...")
    
    # Check for enum values
    required_modes = ['BASIC', 'ADVANCED', 'PROFOUND', 'AGENTIC']
    for mode in required_modes:
        assert f'{mode} = "{mode.lower()}"' in content, f"Missing {mode}"
        print(f"    ✅ RAGMode.{mode}")
    
    print("✅ RAGMode enum has all values\n")


def test_mode_parameters():
    """Test that modes have expected parameters."""
    modes_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'perspicacite', 'rag', 'modes')
    
    print("  Checking mode parameters...")
    
    # Basic mode
    with open(os.path.join(modes_dir, 'basic.py'), 'r') as f:
        basic_content = f.read()
    assert 'initial_docs' in basic_content
    assert 'final_max_docs' in basic_content
    print("    ✅ BasicRAGMode has document limits")
    
    # Advanced mode
    with open(os.path.join(modes_dir, 'advanced.py'), 'r') as f:
        advanced_content = f.read()
    assert 'rephrases' in advanced_content
    assert 'wrrf_k' in advanced_content
    assert 'use_refinement' in advanced_content
    print("    ✅ AdvancedRAGMode has WRRF and refinement settings")
    
    # Profound mode
    with open(os.path.join(modes_dir, 'profound.py'), 'r') as f:
        profound_content = f.read()
    assert 'max_cycles' in profound_content
    assert 'early_exit_confidence' in profound_content
    print("    ✅ ProfoundRAGMode has cycle and exit settings")
    
    # Agentic mode
    with open(os.path.join(modes_dir, 'agentic.py'), 'r') as f:
        agentic_content = f.read()
    assert 'max_iterations' in agentic_content
    assert 'early_exit_confidence' in agentic_content
    print("    ✅ AgenticRAGMode has iteration settings")
    
    print("✅ All modes have expected parameters\n")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG Modes Comprehensive Tests")
    print("="*50 + "\n")
    
    test_rag_modes_files_exist()
    test_mode_class_structure()
    test_prompts_file()
    test_hybrid_retrieval()
    test_config_schema()
    test_engine_imports_all_modes()
    test_rag_model_enum()
    test_mode_parameters()
    
    print("="*50)
    print("All RAG mode tests passed!")
    print("="*50 + "\n")
