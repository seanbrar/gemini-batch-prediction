"""
Comprehensive unit tests for gemini_batch.prompts module
"""

from pydantic import BaseModel
import pytest

from gemini_batch.prompts.batch_prompt_builder import BatchPromptBuilder
from gemini_batch.prompts.structured_prompt_builder import StructuredPromptBuilder


@pytest.mark.unit
class TestBatchPromptBuilder:
    """Test batch prompt builder functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.builder = BatchPromptBuilder()

    def test_create_prompt_single_question(self):
        """Should create prompt for single question"""
        questions = ["What is machine learning?"]
        prompt = self.builder.create_prompt(questions)

        assert "Please answer each of the following questions." in prompt
        assert "Question 1: What is machine learning?" in prompt
        assert "JSON array of strings" in prompt
        assert "Do not include any other text" in prompt

    def test_create_prompt_multiple_questions(self):
        """Should create prompt for multiple questions"""
        questions = [
            "What is AI?",
            "What is machine learning?",
            "What is deep learning?",
        ]
        prompt = self.builder.create_prompt(questions)

        assert "Please answer each of the following questions." in prompt
        assert "Question 1: What is AI?" in prompt
        assert "Question 2: What is machine learning?" in prompt
        assert "Question 3: What is deep learning?" in prompt
        assert "JSON array of strings" in prompt

    def test_create_prompt_empty_questions(self):
        """Should handle empty questions list"""
        questions = []
        prompt = self.builder.create_prompt(questions)

        assert "Please answer each of the following questions." in prompt
        assert "JSON array of strings" in prompt
        # Should not have any question lines
        assert "Question 1:" not in prompt

    def test_create_prompt_question_formatting(self):
        """Should format questions with proper numbering"""
        questions = ["Q1", "Q2", "Q3"]
        prompt = self.builder.create_prompt(questions)

        lines = prompt.split("\n")
        question_lines = [line for line in lines if line.startswith("Question")]

        assert len(question_lines) == 3
        assert question_lines[0] == "Question 1: Q1"
        assert question_lines[1] == "Question 2: Q2"
        assert question_lines[2] == "Question 3: Q3"

    def test_create_prompt_json_instruction(self):
        """Should include proper JSON formatting instructions"""
        questions = ["What is AI?"]
        prompt = self.builder.create_prompt(questions)

        assert "MUST be a single, valid JSON array of strings" in prompt
        assert "corresponding order" in prompt
        assert "Do not include any other text, formatting, or explanations" in prompt

    def test_create_prompt_special_characters(self):
        """Should handle questions with special characters"""
        questions = [
            "What is AI?",
            "What's the difference between ML & DL?",
            "How does NLP work?",
        ]
        prompt = self.builder.create_prompt(questions)

        assert "Question 1: What is AI?" in prompt
        assert "Question 2: What's the difference between ML & DL?" in prompt
        assert "Question 3: How does NLP work?" in prompt


@pytest.mark.unit
class TestStructuredPromptBuilder:
    """Test structured prompt builder functionality"""

    def setup_method(self):
        """Set up test fixtures"""

        class TestSchema(BaseModel):
            answers: list[str]

        self.schema = TestSchema
        self.builder = StructuredPromptBuilder(self.schema)

    def test_create_prompt_single_question(self):
        """Should create prompt for single question with schema"""
        questions = ["What is machine learning?"]
        prompt = self.builder.create_prompt(questions)

        assert (
            "answer the following questions by generating a single, valid JSON object"
            in prompt
        )
        assert "strictly conforms to the provided schema" in prompt
        assert "Question 1: What is machine learning?" in prompt
        assert "Do not include markdown formatting" in prompt

    def test_create_prompt_multiple_questions(self):
        """Should create prompt for multiple questions with schema"""
        questions = [
            "What is AI?",
            "What is machine learning?",
            "What is deep learning?",
        ]
        prompt = self.builder.create_prompt(questions)

        assert (
            "answer the following questions by generating a single, valid JSON object"
            in prompt
        )
        assert "strictly conforms to the provided schema" in prompt
        assert "Question 1: What is AI?" in prompt
        assert "Question 2: What is machine learning?" in prompt
        assert "Question 3: What is deep learning?" in prompt

    def test_create_prompt_empty_questions(self):
        """Should handle empty questions list with schema"""
        questions = []
        prompt = self.builder.create_prompt(questions)

        assert (
            "answer the following questions by generating a single, valid JSON object"
            in prompt
        )
        assert "strictly conforms to the provided schema" in prompt
        # Should not have any question lines
        assert "Question 1:" not in prompt

    def test_create_prompt_schema_instruction(self):
        """Should include proper schema instruction"""
        questions = ["What is AI?"]
        prompt = self.builder.create_prompt(questions)

        assert "single, valid JSON object" in prompt
        assert "strictly conforms to the provided schema" in prompt
        assert (
            "Do not include markdown formatting, explanations, or any other text"
            in prompt
        )

    def test_create_prompt_question_formatting(self):
        """Should format questions with proper numbering"""
        questions = ["Q1", "Q2", "Q3"]
        prompt = self.builder.create_prompt(questions)

        lines = prompt.split("\n")
        question_lines = [line for line in lines if line.startswith("Question")]

        assert len(question_lines) == 3
        assert question_lines[0] == "Question 1: Q1"
        assert question_lines[1] == "Question 2: Q2"
        assert question_lines[2] == "Question 3: Q3"

    def test_create_prompt_special_characters(self):
        """Should handle questions with special characters"""
        questions = [
            "What is AI?",
            "What's the difference between ML & DL?",
            "How does NLP work?",
        ]
        prompt = self.builder.create_prompt(questions)

        assert "Question 1: What is AI?" in prompt
        assert "Question 2: What's the difference between ML & DL?" in prompt
        assert "Question 3: How does NLP work?" in prompt

    def test_create_prompt_different_schema(self):
        """Should work with different schema types"""

        class DifferentSchema(BaseModel):
            result: str
            confidence: float

        builder = StructuredPromptBuilder(DifferentSchema)
        questions = ["What is AI?"]
        prompt = builder.create_prompt(questions)

        assert (
            "answer the following questions by generating a single, valid JSON object"
            in prompt
        )
        assert "strictly conforms to the provided schema" in prompt
        assert "Question 1: What is AI?" in prompt


@pytest.mark.unit
class TestPromptBuilderIntegration:
    """Test prompt builders working together"""

    def test_batch_vs_structured_differences(self):
        """Should show key differences between batch and structured builders"""

        class TestSchema(BaseModel):
            answers: list[str]

        batch_builder = BatchPromptBuilder()
        structured_builder = StructuredPromptBuilder(TestSchema)

        questions = ["What is AI?", "What is ML?"]

        batch_prompt = batch_builder.create_prompt(questions)
        structured_prompt = structured_builder.create_prompt(questions)

        # Batch focuses on JSON array
        assert "JSON array of strings" in batch_prompt
        assert "corresponding order" in batch_prompt

        # Structured focuses on schema compliance
        assert "JSON object" in structured_prompt
        assert "strictly conforms to the provided schema" in structured_prompt

        # Both should have the same question formatting
        assert "Question 1: What is AI?" in batch_prompt
        assert "Question 1: What is AI?" in structured_prompt
        assert "Question 2: What is ML?" in batch_prompt
        assert "Question 2: What is ML?" in structured_prompt

    def test_prompt_builder_consistency(self):
        """Should maintain consistent formatting across builders"""
        questions = ["What is AI?"]

        batch_builder = BatchPromptBuilder()

        class TestSchema(BaseModel):
            answers: list[str]

        structured_builder = StructuredPromptBuilder(TestSchema)

        batch_prompt = batch_builder.create_prompt(questions)
        structured_prompt = structured_builder.create_prompt(questions)

        # Both should start with instruction
        assert batch_prompt.startswith("Please answer")
        assert structured_prompt.startswith("Please answer")

        # Both should have numbered questions
        assert "Question 1: What is AI?" in batch_prompt
        assert "Question 1: What is AI?" in structured_prompt

        # Both should have clear output instructions
        assert "MUST be" in batch_prompt or "Do not include" in batch_prompt
        assert "Do not include" in structured_prompt

    def test_prompt_builder_edge_cases(self):
        """Should handle edge cases gracefully"""
        # Empty questions
        batch_builder = BatchPromptBuilder()

        class TestSchema(BaseModel):
            answers: list[str]

        structured_builder = StructuredPromptBuilder(TestSchema)

        empty_batch = batch_builder.create_prompt([])
        empty_structured = structured_builder.create_prompt([])

        assert "Please answer each of the following questions." in empty_batch
        assert "answer the following questions by generating" in empty_structured
        assert "Question 1:" not in empty_batch
        assert "Question 1:" not in empty_structured

        # Long questions
        long_questions = [
            "This is a very long question that contains many words and should be handled properly by the prompt builder without any issues or truncation",
            "Another long question with special characters: !@#$%^&*() and numbers 1234567890",
        ]

        long_batch = batch_builder.create_prompt(long_questions)
        long_structured = structured_builder.create_prompt(long_questions)

        assert "Question 1: This is a very long question" in long_batch
        assert "Question 2: Another long question with special characters" in long_batch
        assert "Question 1: This is a very long question" in long_structured
        assert (
            "Question 2: Another long question with special characters"
            in long_structured
        )
