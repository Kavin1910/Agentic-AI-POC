import streamlit as st
import os
import toml
from typing import List, Dict
from dataclasses import dataclass
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.googlesearch import GoogleSearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration handling
def load_config():
    """Load configuration from multiple possible sources"""
    config = {
        "QDRANT_URL": "API",  # default values
        "QDRANT_API_KEY": "API"
    }
    
    # Try loading from secrets.toml in current directory
    try:
        with open("secrets.toml", "r") as f:
            toml_config = toml.load(f)
            config.update(toml_config)
    except FileNotFoundError:
        st.warning("secrets.toml not found in current directory. Using default or environment variables.")
    
    # Override with environment variables if they exist
    for key in config:
        env_value = os.getenv(key)
        if env_value:
            config[key] = env_value
    
    return config

@dataclass
class AgentResponse:
    source: str
    content: str
    confidence: float = 1.0

class AyurvedaResearchAgent:
    def __init__(self):
        self.agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile"),
            tools=[GoogleSearch()],
            description="Expert Ayurveda research assistant",
            instructions=[
                "Search for authoritative information about Ayurvedic topics",
                "Focus on scientific validation and clinical studies",
                "Provide detailed analysis with references",
            ],
            show_tool_calls=True
        )
    
    def process(self, query: str) -> AgentResponse:
        try:
            response = self.agent.run(f"Research the following Ayurvedic topic and provide detailed information: {query}")
            return AgentResponse(
                source="Research",
                content=response.content
            )
        except Exception as e:
            st.error(f"Research Agent Error: {str(e)}")
            return AgentResponse(
                source="Research",
                content="Unable to process research query at this time."
            )

class AyurvedaDocumentAgent:
    def __init__(self, api_url: str = None, api_key: str = None):
        try:
            if api_url and api_key:
                self.client = QdrantClient(url=api_url, api_key=api_key)
            else:
                self.client = QdrantClient(":memory:")  # Use in-memory storage for testing
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.collection_name = "myayurhealth_docs"
        except Exception as e:
            st.error(f"Document Agent Initialization Error: {str(e)}")
            self.client = None
            self.model = None

    def process(self, query: str) -> List[AgentResponse]:
        if not self.client or not self.model:
            return [AgentResponse(
                source="Documentation",
                content="Document retrieval system is not available.",
                confidence=0.0
            )]
        
        try:
            query_vector = self.model.encode(query).tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            return [
                AgentResponse(
                    source="Documentation",
                    content=result.payload['text'],
                    confidence=float(result.score)
                )
                for result in results
            ]
        except Exception as e:
            st.error(f"Document Search Error: {str(e)}")
            return [AgentResponse(
                source="Documentation",
                content="Unable to search documents at this time.",
                confidence=0.0
            )]

class AyurvedaGenerationAgent:
    def __init__(self):
        self.agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile"),
            description="Ayurvedic knowledge synthesis specialist",
            instructions=[
                "Synthesize information from multiple sources",
                "Provide practical, actionable insights",
                "Maintain authenticity of Ayurvedic principles",
                "Create clear, structured responses"
            ]
        )
    
    def process(self, query: str, context: List[str]) -> AgentResponse:
        try:
            combined_prompt = f"""
            Query: {query}
            
            Context Information:
            {' '.join(context)}
            
            Based on the above information, provide a comprehensive, well-structured response that:
            1. Synthesizes key points from all sources
            2. Highlights practical applications
            3. Notes any areas of consensus or disagreement
            4. Provides actionable recommendations
            """
            
            response = self.agent.run(combined_prompt)
            return AgentResponse(
                source="Synthesis",
                content=response.content
            )
        except Exception as e:
            st.error(f"Generation Agent Error: {str(e)}")
            return AgentResponse(
                source="Synthesis",
                content="Unable to generate synthesis at this time."
            )

class IntegratedAyurvedaAssistant:
    def __init__(self, config: Dict[str, str]):
        self.research_agent = AyurvedaResearchAgent()
        self.document_agent = AyurvedaDocumentAgent(
            api_url=config.get("QDRANT_URL"),
            api_key=config.get("QDRANT_API_KEY")
        )
        self.generation_agent = AyurvedaGenerationAgent()
    
    def process_query(self, query: str) -> Dict[str, AgentResponse]:
        research_response = self.research_agent.process(query)
        doc_responses = self.document_agent.process(query)
        
        context = [
            research_response.content,
            *[resp.content for resp in doc_responses]
        ]
        
        synthesis_response = self.generation_agent.process(query, context)
        
        return {
            "research": research_response,
            "documentation": doc_responses,
            "synthesis": synthesis_response
        }

def display_response(response: AgentResponse, container):
    container.markdown(f"**Confidence Score:** {response.confidence:.2f}" if response.confidence != 1.0 else "")
    container.markdown(response.content)

def main():
    st.set_page_config(page_title="Integrated Ayurveda Assistant", layout="wide")
    st.title("Integrated Ayurveda Assistant")

    # Load configuration
    config = load_config()

    # Initialize the assistant
    assistant = IntegratedAyurvedaAssistant(config)

    # Query input
    query = st.text_input("Enter your Ayurvedic health query:")

    if st.button("Submit"):
        if not query:
            st.warning("Please enter a query.")
            return

        with st.spinner("Processing your query..."):
            results = assistant.process_query(query)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Research Findings")
                display_response(results["research"], st)
            
            with col2:
                st.markdown("### Documentation Matches")
                for idx, doc_response in enumerate(results["documentation"], 1):
                    with st.expander(f"Document Match {idx}"):
                        display_response(doc_response, st)

            st.markdown("### 🔍 Comprehensive Analysis")
            st.info("This analysis combines insights from all sources")
            display_response(results["synthesis"], st)

if __name__ == "__main__":
    main()
