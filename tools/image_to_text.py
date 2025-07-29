from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.base_language import BaseLanguageModel
import base64
from PIL import Image
import os

class ImageToTextInput(BaseModel):
    """Input parameter model"""
    image_path: str = Field(..., description="Image file path")
    prompt: str = Field(default="Please describe this image", description="Prompt text")

class ImageToTextTool(BaseTool):
    name = "image_to_text"
    description = "Tool for converting images to text descriptions"
    args_schema: type[BaseModel] = ImageToTextInput
    
    llm: BaseLanguageModel = Field(description="Language Model to use")
    
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm=llm)
        if not llm:
            raise ValueError("Must provide LLM instance")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _run(self, image_path: str, prompt: str = "Please describe this image") -> str:
        """Run tool"""
        try:
            if not os.path.exists(image_path):
                return "Error: Image file does not exist"
            try:
                Image.open(image_path)
            except:
                return "Error: File is not a valid image format"

            base64_image = self._encode_image(image_path)

            messages = [
                {"role": "system", "content": "You are a professional data analyst."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error: Image processing failed - {str(e)}"