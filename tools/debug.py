from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.base_language import BaseLanguageModel
import re
from typing import Dict, List, Optional, Union
from prompts.debug_prompts import (
    DEBUG_TEMPLATE,
    DEBUG_LOCATE_TEMPLATE,
    DEBUG_FIX_TEMPLATE, 
    DEBUG_MERGE_TEMPLATE,
    DEBUG_ASK_FOR_HELP_TEMPLATE,
    DEBUG_TEST_LOCATE_TEMPLATE,
    DEBUG_TEST_FIX_TEMPLATE,
    DEBUG_TEST_MERGE_TEMPLATE
)
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class DebugInput(BaseModel):
    """Input parameter model"""
    code: str = Field(..., description="Code to debug")
    error_message: str = Field(..., description="Error message")
    output_message: str = Field("", description="Code output message (optional)")
    tools_description: str = Field("", description="Tool description (optional)")
    is_test_error: bool = Field(False, description="Whether it's a test error")
    test_info: str = Field("", description="Test related information")

class DebugTool(BaseTool):
    name = "debug"
    description = "Code debugging tool"
    args_schema: type[BaseModel] = DebugInput
    
    llm: BaseLanguageModel = Field(description="Language Model to use")
    all_error_messages: List[str] = Field(default_factory=list, description="List to store all error messages")
    memory: ConversationBufferMemory = Field(default_factory=lambda: ConversationBufferMemory(), description="Conversation memory")
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize tool"""
        super().__init__(llm=llm)
        if not llm:
            raise ValueError("Must provide LLM instance")
        self.memory = ConversationBufferMemory()
        # Add system message
        system_message = SystemMessage(content="You are a professional code debugging assistant, skilled at analyzing and fixing code errors. Please help users locate and solve problems in their code.")
        self.memory.chat_memory.add_message(system_message)
            
    def _chat_with_memory(self, prompt: str) -> str:
        """Chat with memory"""
        self.memory.chat_memory.add_message(HumanMessage(content=prompt))
        chat_prompt = ChatPromptTemplate.from_messages(self.memory.chat_memory.messages)
        chain = chat_prompt | self.llm
        response = chain.invoke({})
        reply = response.content
        self.memory.chat_memory.add_message(AIMessage(content=reply))
        return reply

    def _locate_error(self, wrong_code: str, error_messages: str, output_messages: str) -> str:
        """Locate error"""
        prompt = DEBUG_LOCATE_TEMPLATE.format(
            wrong_code=wrong_code,
            error_messages=error_messages,
            output_messages=output_messages,
        )
        prompt = DEBUG_TEMPLATE + prompt
        response = self._chat_with_memory(prompt)
        return self._extract_code(response)
        
    def _fix_error(self, error_code: str, error_messages: str, output_messages: str) -> str:
        """Fix error"""
        prompt = DEBUG_FIX_TEMPLATE.format(
            most_relevant_code_snippet=error_code,
            error_messages=error_messages,
            output_messages=output_messages,
        )
        response = self._chat_with_memory(prompt)
        return self._extract_code(response)

    def _merge_code(self, wrong_code: str, error_snippet: str, fixed_snippet: str) -> str:
        """Merge fixed code"""
        prompt = DEBUG_MERGE_TEMPLATE.format(
            wrong_code=wrong_code,
            most_relevant_code_snippet=error_snippet,
            code_snippet_after_correction=fixed_snippet
        )
        response = self._chat_with_memory(prompt)
        return self._extract_code(response)

    def _check_need_help(self, error_message: str) -> bool:
        """Check if help is needed"""
        self.all_error_messages.append(error_message)
        if len(self.all_error_messages) >= 3:
            prompt = DEBUG_ASK_FOR_HELP_TEMPLATE.format(
                i=len(self.all_error_messages),
                all_error_messages="\n".join(self.all_error_messages)
            )
            response = self._chat_with_memory(prompt)
            return "<HELP>" in response
        return False

    def _extract_code(self, text: str) -> str:
        """Extract code block from text"""
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1] if matches else ""

    def _debug_test_failure(self, code: str, test_info: str, output_message: str = "") -> Dict:
        """Handle test failure cases"""
        try:
            # 1. Locate test failure code snippet
            error_snippet = self._locate_test_error(
                code,
                test_info,
                output_message
            )
            
            # 2. Fix test issue
            fixed_snippet = self._fix_test_error(
                error_snippet,
                test_info,
                output_message
            )
            
            # 3. Merge fixed code
            fixed_code = self._merge_test_fix(code, error_snippet, fixed_snippet)
            
            return {
                "status": "success",
                "fixed_code": fixed_code
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during test debugging: {str(e)}"
            }

    def _locate_test_error(self, code: str,test_info: str, output_message: str) -> str:
        """Locate test failure code location"""
        prompt = DEBUG_TEST_LOCATE_TEMPLATE.format(
            code=code,
            test_info=test_info,
            output_message=output_message
        )
        response = self._chat_with_memory(prompt)
        return self._extract_code(response)

    def _fix_test_error(self, error_snippet: str, test_info: str, output_message: str) -> str:
        """Fix test failure code"""
        prompt = DEBUG_TEST_FIX_TEMPLATE.format(
            error_snippet=error_snippet,
            test_info=test_info,
            output_message=output_message
        )
        response = self._chat_with_memory(prompt)
        return self._extract_code(response)

    def _merge_test_fix(self, original_code: str, error_snippet: str, fixed_snippet: str) -> str:
        """Merge test fixed code"""
        prompt = DEBUG_TEST_MERGE_TEMPLATE.format(
            original_code=original_code,
            error_snippet=error_snippet,
            fixed_snippet=fixed_snippet
        )
        response = self._chat_with_memory(prompt)
        return self._extract_code(response)

    def _debug_runtime_error(self, code: str, error_message: str, output_message: str = "") -> Dict:
        """Handle runtime errors"""
        try:
            # 1. Locate error
            error_snippet = self._locate_error(
                code, 
                error_message, 
                output_message, 
            )
            
            # 2. Fix error
            fixed_snippet = self._fix_error(
                error_snippet, 
                error_message, 
                output_message
            )
            
            # 3. Merge code
            fixed_code = self._merge_code(code, error_snippet, fixed_snippet)
            
            return {
                "status": "success",
                "fixed_code": fixed_code
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during debugging process: {str(e)}"
            }

    def _run(self, code: str, error_message: str, output_message: str = "", 
             tools_description: str = "", 
             is_test_error: bool = False) -> Dict:
        """Run debugging tool"""
        try:
            # Check if help is needed
            if self._check_need_help(error_message):
                return {
                    "status": "need_help",
                    "message": "Human assistance needed"
                }
            
            # Choose different debugging strategy based on error type
            if is_test_error:
                return self._debug_test_failure(
                    code,
                    error_message,
                    output_message
                )
            else:
                # Original error debugging logic
                return self._debug_runtime_error(
                    code,
                    error_message,
                    output_message,
                )
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during debugging process: {str(e)}"
            } 