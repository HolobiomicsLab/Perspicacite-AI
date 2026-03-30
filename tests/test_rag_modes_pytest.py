"""Pytest-compatible tests for RAG modes.

These tests verify the structure and existence of RAG mode components
without requiring external dependencies like chromadb.
"""

import os
import pytest


class TestRAGModesStructure:
    """Test RAG mode file structure."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return os.path.join(os.path.dirname(__file__), '..')
    
    @pytest.fixture
    def modes_dir(self, project_root):
        """Get modes directory."""
        return os.path.join(project_root, 'src', 'perspicacite', 'rag', 'modes')
    
    def test_all_mode_files_exist(self, modes_dir):
        """Test that all RAG mode files exist."""
        required_files = [
            'base.py', 'basic.py', 'advanced.py', 
            'profound.py', 'agentic.py', '__init__.py'
        ]
        
        for file in required_files:
            filepath = os.path.join(modes_dir, file)
            assert os.path.exists(filepath), f"Missing {file}"
    
    @pytest.mark.parametrize("file,methods", [
        ('basic.py', ['execute', 'execute_stream', '_generate_response']),
        ('advanced.py', ['execute', 'execute_stream', '_generate_similar_queries', '_wrrf_retrieval']),
        ('profound.py', ['execute', 'execute_stream', '_create_plan', '_execute_step']),
        ('agentic.py', ['execute', 'execute_stream']),  # Now a wrapper around AgenticOrchestrator
    ])
    def test_mode_has_required_methods(self, modes_dir, file, methods):
        """Test that each mode has required methods."""
        filepath = os.path.join(modes_dir, file)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        for method in methods:
            assert f'def {method}' in content, f"{file} missing {method}"


class TestPromptsStructure:
    """Test prompts file structure."""
    
    @pytest.fixture
    def prompts_path(self):
        """Get prompts file path."""
        return os.path.join(
            os.path.dirname(__file__), '..',
            'src', 'perspicacite', 'rag', 'prompts.py'
        )
    
    def test_prompts_file_exists(self, prompts_path):
        """Test that prompts.py exists."""
        assert os.path.exists(prompts_path)
    
    @pytest.mark.parametrize("prompt_name", [
        'MANDATORY_PROMPT',
        'DEFAULT_SYSTEM_PROMPT',
        'FORMAT_PROMPT',
        'GENERATE_SIMILAR_QUERIES_PROMPT',
        'EVALUATE_RESPONSE_PROMPT',
        'PROFOUND_CREATE_PLAN_PROMPT',
        'PROFOUND_IS_QUESTION_ANSWERED_PROMPT',
    ])
    def test_prompt_exists(self, prompts_path, prompt_name):
        """Test that each required prompt exists."""
        with open(prompts_path, 'r') as f:
            content = f.read()
        
        assert prompt_name in content, f"Missing {prompt_name}"


class TestHybridRetrieval:
    """Test hybrid retrieval module."""
    
    @pytest.fixture
    def hybrid_path(self):
        """Get hybrid.py path."""
        return os.path.join(
            os.path.dirname(__file__), '..',
            'src', 'perspicacite', 'retrieval', 'hybrid.py'
        )
    
    def test_hybrid_file_exists(self, hybrid_path):
        """Test that hybrid.py exists."""
        assert os.path.exists(hybrid_path)
    
    @pytest.mark.parametrize("func", [
        'normalize_scores',
        'combine_scores',
        'compute_bm25_scores',
        'hybrid_retrieval',
    ])
    def test_function_exists(self, hybrid_path, func):
        """Test that each required function exists."""
        with open(hybrid_path, 'r') as f:
            content = f.read()
        
        assert f'def {func}' in content, f"Missing {func}"


class TestConfigSchema:
    """Test configuration schema."""
    
    @pytest.fixture
    def config_path(self):
        """Get config schema path."""
        return os.path.join(
            os.path.dirname(__file__), '..',
            'src', 'perspicacite', 'config', 'schema.py'
        )
    
    def test_rag_modes_config_exists(self, config_path):
        """Test that RAGModesConfig class exists."""
        with open(config_path, 'r') as f:
            content = f.read()
        
        assert 'class RAGModesConfig' in content
    
    @pytest.mark.parametrize("mode", ['basic', 'advanced', 'profound', 'agentic'])
    def test_mode_config_exists(self, config_path, mode):
        """Test that each mode has config settings."""
        with open(config_path, 'r') as f:
            content = f.read()
        
        assert f'{mode}:' in content, f"Missing {mode} config"


class TestRAGEngine:
    """Test RAG engine."""
    
    @pytest.fixture
    def engine_path(self):
        """Get engine.py path."""
        return os.path.join(
            os.path.dirname(__file__), '..',
            'src', 'perspicacite', 'rag', 'engine.py'
        )
    
    @pytest.mark.parametrize("mode_class", [
        'BasicRAGMode',
        'AdvancedRAGMode',
        'ProfoundRAGMode',
        'AgenticRAGMode',
    ])
    def test_engine_imports_mode(self, engine_path, mode_class):
        """Test that engine imports each mode class."""
        with open(engine_path, 'r') as f:
            content = f.read()
        
        assert mode_class in content, f"Missing {mode_class}"
    
    @pytest.mark.parametrize("mode", ['BASIC', 'ADVANCED', 'PROFOUND', 'AGENTIC'])
    def test_engine_registers_mode(self, engine_path, mode):
        """Test that engine registers each mode."""
        with open(engine_path, 'r') as f:
            content = f.read()
        
        assert f'RAGMode.{mode}' in content, f"Missing RAGMode.{mode}"


class TestRAGModel:
    """Test RAG model."""
    
    @pytest.fixture
    def rag_model_path(self):
        """Get rag.py path."""
        return os.path.join(
            os.path.dirname(__file__), '..',
            'src', 'perspicacite', 'models', 'rag.py'
        )
    
    @pytest.mark.parametrize("mode", ['BASIC', 'ADVANCED', 'PROFOUND', 'AGENTIC'])
    def test_rag_mode_enum_value(self, rag_model_path, mode):
        """Test that RAGMode enum has all values."""
        with open(rag_model_path, 'r') as f:
            content = f.read()
        
        assert f'{mode} = "{mode.lower()}"' in content, f"Missing RAGMode.{mode}"
