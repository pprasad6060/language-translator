from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

class GroqModule:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.7):
        self.model = model_name
        self.output_parser = StrOutputParser()
        self.temperature = temperature
        load_dotenv()

    def load_api_key(self)-> str:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
          raise ValueError("GROQ_API_KEY environment variable not set")
        return groq_api_key
    
    def get_groq_model(self) -> ChatGroq:
      groq_api_key = self.load_api_key()
      groq_model = ChatGroq(model = self.model, temperature = self.temperature,
       groq_api_key = groq_api_key)
      return groq_model

    def get_prompt_template(self) -> ChatPromptTemplate:
      system_template = "Act as a highly linguistically accurate translator and translate below to {target_language}:"
      prompt_template = ChatPromptTemplate.from_messages([("system", system_template),
                                                        ("user", '{user_text}')])
      return prompt_template

    def get_chain(self):
      groq_model = self.get_groq_model()
      prompt_template = self.get_prompt_template()
      return prompt_template|groq_model|self.output_parser
    

