from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json
from langchain.tools import tool
import re
from tools import *
import os
import logging
from datetime import datetime
import time
import openai

class BaseDSLC_Agent():
    def __init__(
        self, 
        name: str,
        system_message: str,
        memory=None,
        llm=None,
        tools=None,
        max_turns=3
    ):
        self.name = name
        self.memory = memory or ConversationBufferMemory()
        self.llm = llm
        self.max_turns = max_turns
        
        self.system_message = SystemMessage(content=system_message)
        self.memory.chat_memory.add_message(self.system_message)
        
        self.tools = tools or []
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
        self.setup_logger()
        self.logger.info(f"Initialized {self.name} agent")

    def setup_logger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.info(f"Logger setup completed for {self.name}")

    def get_memory_token(self):
        buffer = self.memory.chat_memory.messages
        num_tokens = self.llm.get_num_tokens_from_messages(buffer)
        return num_tokens

    def get_recent_k_conversations(self):
        """Get the last k rounds of conversations"""
        return self.memory.chat_memory.messages[-self.max_turns:]

    def execute(self, input_data):
        """Execute agent's task and generate response through SystemMessage and HumanMessage using LLM"""
        try:
            human_message = HumanMessage(content=input_data)
            ai_response = self.conversation.predict(input=human_message.content)
            self.logger.info(f"Execution completed successfully")
            return ai_response
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            raise
    
    def execute2(self, input_data):
        """Execute agent's task with controlled conversation turns"""
        try:
            human_message = HumanMessage(content=input_data)
            self.memory.chat_memory.add_message(human_message)
            recent_messages = self.get_recent_k_conversations()

            prompt = ChatPromptTemplate.from_messages(recent_messages)
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)

            ai_response = chain.run(input_data=input_data)
            self.memory.chat_memory.add_message(AIMessage(content=ai_response))
            self.logger.info("Execution with controlled turns completed successfully")
            return ai_response
        except Exception as e:
            self.logger.error(f"Error during controlled execution: {str(e)}")
            raise
    
    def chat_with_memory(self, prompt: str, memory: ConversationBufferMemory):
        """Chat with memory"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if not any(isinstance(msg, SystemMessage) for msg in memory.chat_memory.messages):
                    memory.chat_memory.add_message(self.system_message)
                memory.chat_memory.add_message(HumanMessage(content=prompt))
                chat_prompt = ChatPromptTemplate.from_messages(memory.chat_memory.messages)
                chain = chat_prompt | self.llm
                response = chain.invoke({})
                reply = response.content
                memory.chat_memory.add_message(AIMessage(content=reply))
                return reply, memory
                
            except openai.AuthenticationError as e:
                if "无效的令牌" in str(e):
                    if attempt == max_retries - 1:
                        self.logger.error(f"Authentication failed after {max_retries} attempts: {str(e)}")
                        raise
                    else:
                        self.logger.warning(f"Token error on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                        if memory.chat_memory.messages:
                            memory.chat_memory.messages.pop()
                        time.sleep(retry_delay)
                        continue
                else:
                    raise
                    
            except Exception as e:
                self.logger.error(f"Error during chat with memory: {str(e)}")
                raise
    
    def parse_llm_json(self, response):
        """
        Parse JSON data from LLM response using regular expressions.
        """
        try:
            code_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if code_match:
                json_str = code_match.group(1).strip()
                return json.loads(json_str)
            return f"No JSON data found, original response: {response}"
        except json.JSONDecodeError as e:
            return f"JSON parsing error: {str(e)}, original response: {response}"
    
    def save_history_to_txt(self, filepath):
        """Save conversation history to text file, excluding SystemMessage"""
        messages = self.memory.chat_memory.messages
        filtered_messages = [
            {"type": type(msg).__name__, "content": msg.content}
            for msg in messages if not isinstance(msg, SystemMessage)
        ]
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(filtered_messages, f, ensure_ascii=False, indent=4)
            return f"Conversation history saved to {filepath}"
        except Exception as e:
            return f"Save failed: {str(e)}"

    def load_history_from_txt(self, filepath):
        """Load conversation history from text file to Agent"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)
            for msg in loaded_messages:
                if msg["type"] == "HumanMessage":
                    self.memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
                elif msg["type"] == "AIMessage":
                    self.memory.chat_memory.add_message(AIMessage(content=msg["content"]))
            return f"History loaded from {filepath}"
        except Exception as e:
            return f"Load failed: {str(e)}"
        
    def execute_generated_code(self, generated_code: str, save_path: str = None, is_debug: bool = False, retry_count: int = 0, max_retries: int = 3) -> str:
        """Execute Python code generated by LLM
        
        Args:
            generated_code: Generated code string
            save_path: Path to save the code
            is_debug: Whether in debug mode
            retry_count: Current retry count
            max_retries: Maximum retry attempts
        """
        self.logger.info(f"Executing generated code{' (debug mode)' if is_debug else ''} - Attempt {retry_count + 1}/{max_retries}")
        
        if retry_count >= max_retries:
            self.logger.warning(f"Reached maximum retry attempts ({max_retries})")
            return "Code execution failed, maximum retries reached. Need to regenerate code."
            
        code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
        if not code_match:
            self.logger.error("Failed to extract code from generated text")
            return "Cannot extract generated code, please check code format."
        
        extracted_code = code_match.group(1)
        
        if not is_debug:
            path_setup_code = '''import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
            extracted_code = path_setup_code + extracted_code
        
        output = []
        original_print = print
        def capture_print(*args, **kwargs):
            output.append(' '.join(map(str, args)))
        
        try:
            globals()['print'] = capture_print
            exec(extracted_code, globals())
            globals()['print'] = original_print
            
            execution_output = '\n'.join(output)
            if 'result' in globals() and globals()['result'] is not None:
                result = str(globals()['result'])
            elif execution_output:
                result = execution_output
            else:
                result = "Code execution successful, but no output result."
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                code_to_save = extracted_code if is_debug else code_match.group(1)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(code_to_save)
                self.logger.info(f"Successful code saved to {save_path}")
            
            self.logger.info("Code execution completed successfully")
            return result
            
        except Exception as e:
            globals()['print'] = original_print
            self.logger.error(f"Error during code execution: {str(e)}")
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.extract_tb(exc_traceback)
            
            error_info = []
            code_lines = extracted_code.splitlines()
            
            for frame in tb:
                if frame.filename == '<string>':
                    line_no = frame.lineno
                    if 1 <= line_no <= len(code_lines):
                        context = code_lines[line_no-1].strip()
                        error_info.append(f"Line {line_no}: {context}")
            
            if error_info:
                error_message = f"Error message: {str(e)}\nError location:\n" + "\nCall: ".join(error_info)
            else:
                error_message = str(e)
                
            output_message = '\n'.join(output)
            
            debug_tool = DebugTool(llm=self.llm)
            #print(extracted_code)
            print(error_message)
            debug_result = debug_tool.run(tool_input={
                "code": extracted_code,
                "error_message": error_message,
                "output_message": output_message,
                "tools_description": "Code debugging tool"
            })
            
            if debug_result["status"] == "need_help":
                return f"Code execution failed, manual debugging required. Error message: {error_message}"
            elif debug_result["status"] == "success":
                return self.execute_generated_code(
                    f"```python\n{debug_result['fixed_code']}\n```",
                    save_path,
                    is_debug=True,
                    retry_count=retry_count + 1,
                    max_retries=max_retries
                )
            else:
                return f"Code execution failed: {error_message}\nDebugging failed: {debug_result['message']}"
            
    def run_unit_tests(self, cleaned_data_path: str, original_data_path: str = None) -> str:
        """
        Run unit tests for data cleaning phase
        
        Args:
            cleaned_data_path: Path to cleaned data file
            original_data_path: Path to original data file (optional)
        
        Returns:
            Tuple[bool, str]: (all tests passed, test report)
        """
        from tools import UnitTestTool
        
        test_tool = UnitTestTool()
        results = test_tool.run(tool_input={
            "phase": "Data Cleaning",
            "data_path": cleaned_data_path,
            "original_data_path": original_data_path
        })
        
        return results["all_passed"], results["report"]