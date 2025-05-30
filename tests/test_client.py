"""
Test client for Agent HTTP API
Example usage of the REST endpoints
"""

import requests
import json
import time
from typing import Dict, Any, Optional
import argparse
import unittest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentAPIClient:
    """Client for interacting with Agent HTTP API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def get_agent_types(self) -> Dict[str, Any]:
        """Get available agent types"""
        response = self.session.get(f"{self.base_url}/agents")
        response.raise_for_status()
        return response.json()
    
    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Get information about specific agent type"""
        response = self.session.get(f"{self.base_url}/agents/{agent_type}")
        response.raise_for_status()
        return response.json()
    
    def query_agent(self, 
                   query: str, 
                   agent_type: str = "advanced",
                   thread_id: Optional[str] = None,
                   parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send query to agent"""
        
        payload = {
            "query": query,
            "agent_type": agent_type,
            "thread_id": thread_id,
            "parameters": parameters or {}
        }
        
        response = self.session.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        return response.json()


class TestAgentAPI(unittest.TestCase):
    """Test cases for Agent API"""
    
    def setUp(self):
        """Set up test client"""
        self.client = AgentAPIClient()
        logger.info("Test client initialized")
    
    def test_health_check(self):
        """Test health check endpoint"""
        logger.info("Testing health check endpoint...")
        result = self.client.health_check()
        logger.info(f"Health check response: {result}")
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'healthy')  # Changed from 'ok' to 'healthy'
    
    def test_get_agent_types(self):
        """Test getting available agent types"""
        logger.info("Testing get agent types endpoint...")
        try:
            result = self.client.get_agent_types()
            logger.info(f"Agent types response: {result}")
            self.assertIn('agent_types', result)
            self.assertIsInstance(result['agent_types'], dict)
            self.assertGreater(len(result['agent_types']), 0)
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
            raise
    
    def test_get_agent_info(self):
        """Test getting agent info"""
        logger.info("Testing get agent info endpoint...")
        result = self.client.get_agent_info('standard')
        logger.info(f"Agent info response: {result}")
        self.assertIn('name', result)
        self.assertIn('description', result)
    
    def test_query_agent(self):
        """Test querying agent"""
        logger.info("Testing query agent endpoint...")
        query = "What is 2+2?"
        result = self.client.query_agent(query, "standard")
        logger.info(f"Query response: {result}")
        self.assertIn('answer', result)
        self.assertIn('metadata', result)
        self.assertIsInstance(result['answer'], str)
        self.assertGreater(len(result['answer']), 0)
    
    def test_language_consistency(self):
        """Test language consistency"""
        logger.info("Testing language consistency...")
        query = "Co to jest sztuczna inteligencja?"
        result = self.client.query_agent(query, "standard")
        logger.info(f"Language test response: {result}")
        self.assertIn('answer', result)
        answer = result['answer'].lower()
        
        # Rozszerzona lista polskich słów kluczowych
        polish_keywords = [
            'sztuczna', 'inteligencja', 'system', 'uczenie', 'maszynowe',
            'algorytm', 'dane', 'komputer', 'program', 'technologia',
            'automatyzacja', 'rozwiązanie', 'problem', 'zadanie'
        ]
        
        found_keywords = [word for word in polish_keywords if word in answer]
        logger.info(f"Found Polish keywords: {found_keywords}")
        
        self.assertTrue(
            len(found_keywords) >= 3,
            f"Odpowiedź powinna zawierać co najmniej 3 polskie słowa kluczowe. Znaleziono: {found_keywords}"
        )


def main():
    """Main test client function"""
    
    parser = argparse.ArgumentParser(description="Agent API Test Client")
    parser.add_argument(
        "--mode", 
        choices=["demo", "agents", "language", "errors", "performance", "threads", "tools", "comprehensive"], 
        default="demo",
        help="Test mode to run"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "demo":
            unittest.main(argv=['first-arg-is-ignored'], exit=False)
        elif args.mode == "comprehensive":
            unittest.main(argv=['first-arg-is-ignored'], exit=False)
        else:
            print(f"Mode {args.mode} not implemented in unittest format")
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    main() 