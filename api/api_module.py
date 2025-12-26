from fastapi import FastAPI
from models.groq_module import GroqModule
from langserve import add_routes
import uvicorn

groqModule = GroqModule()
chain = groqModule.get_chain()

###API Definition
app = FastAPI(title="Langchain Server", version="1.0", description="A simple API using Langchain runnable interfaces")

### Adding chain routes
add_routes(app, chain, path="/translate")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
