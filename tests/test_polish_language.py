"""
Test sprawdzający czy agenty odpowiadają w języku polskim na polskie pytania.
Wymaga skonfigurowanego klucza API OpenAI.
"""

import os
import sys
from unittest import TestCase
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from core.standard_agent import StandardAgent
from core.advanced_agent import AdvancedResearchAgent


class TestPolishLanguage(TestCase):
    """Testy sprawdzające obsługę języka polskiego przez agenty"""
    
    def setUp(self):
        """Przygotowanie testu"""
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY nie jest ustawiony")
            
        print("\n🔧 Konfiguracja testu...")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # Podstawowe narzędzie do wyszukiwania
        self.tools = [DuckDuckGoSearchRun()]
        
        # Pytanie testowe po polsku
        self.polish_query = "Co to jest sztuczna inteligencja?"
        
        # Oczekiwane polskie słowa kluczowe
        self.expected_polish_keywords = [
            "sztuczna inteligencja",
            "system",
            "uczenie",
            "maszynowe",
            "dane",
            "algorytm"
        ]
        print("✅ Konfiguracja zakończona")
    
    def test_standard_agent_polish_response(self):
        """Test czy Standard Agent odpowiada po polsku"""
        print("\n🧪 Test Standard Agent...")
        print(f"📝 Pytanie: {self.polish_query}")
        
        agent = StandardAgent(self.llm, self.tools, verbose=True)
        response = agent.process(self.polish_query, "test_thread")
        
        print(f"\n📄 Otrzymana odpowiedź:\n{response}\n")
        
        # Sprawdź czy odpowiedź zawiera polskie słowa kluczowe
        found_keywords = [word for word in self.expected_polish_keywords 
                         if word.lower() in response.lower()]
        
        print(f"🔍 Znalezione słowa kluczowe: {found_keywords}")
        
        self.assertGreater(
            len(found_keywords),
            2,
            "Odpowiedź powinna zawierać przynajmniej 3 polskie słowa kluczowe"
        )
        
        # Sprawdź czy odpowiedź jest sensownie długa
        self.assertGreater(
            len(response),
            100,
            "Odpowiedź powinna być wystarczająco rozbudowana"
        )
    
    def test_advanced_agent_polish_response(self):
        """Test czy Advanced Agent odpowiada po polsku"""
        print("\n🧪 Test Advanced Agent...")
        print(f"📝 Pytanie: {self.polish_query}")
        
        agent = AdvancedResearchAgent(self.llm, self.tools, verbose=True)
        response = agent.process(self.polish_query, "test_thread")
        
        print(f"\n📄 Otrzymana odpowiedź:\n{response}\n")
        
        # Sprawdź czy odpowiedź zawiera polskie słowa kluczowe
        found_keywords = [word for word in self.expected_polish_keywords 
                         if word.lower() in response.lower()]
        
        print(f"🔍 Znalezione słowa kluczowe: {found_keywords}")
        
        self.assertGreater(
            len(found_keywords),
            2,
            "Odpowiedź powinna zawierać przynajmniej 3 polskie słowa kluczowe"
        )
        
        # Sprawdź czy odpowiedź jest sensownie długa
        self.assertGreater(
            len(response),
            100,
            "Odpowiedź powinna być wystarczająco rozbudowana"
        )


if __name__ == "__main__":
    import unittest
    unittest.main() 