from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class LLMFactory:

    @staticmethod
    def get_llm():

        return ChatGroq(

            api_key=os.getenv(
                "GROQ_API_KEY"
            ),

            model=os.getenv(
                "MODEL_NAME"
            ),

            temperature=0.2
        )


llm=LLMFactory.get_llm()