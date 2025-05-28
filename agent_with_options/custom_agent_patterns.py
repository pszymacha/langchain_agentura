"""
Alternative patterns for advanced agent control
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from enum import Enum

class AgentMode(Enum):
    """Different operational modes for the agent"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    PROBLEM_SOLVING = "problem_solving"

class ControlledAgentExecutor(AgentExecutor):
    """Extended AgentExecutor with custom control logic"""
    
    def __init__(self, 
                 agent, 
                 tools, 
                 mode: AgentMode = AgentMode.RESEARCH,
                 max_research_iterations: int = 3,
                 require_reflection: bool = True,
                 **kwargs):
        super().__init__(agent=agent, tools=tools, **kwargs)
        self.mode = mode
        self.max_research_iterations = max_research_iterations
        self.require_reflection = require_reflection
        self.iteration_count = 0
        self.reflection_history = []
    
    def _get_mode_specific_prompt(self, query: str) -> str:
        """Get mode-specific system prompt"""
        prompts = {
            AgentMode.RESEARCH: f"""
            You are in RESEARCH mode. For this query: "{query}"
            
            MANDATORY WORKFLOW:
            1. ALWAYS start with a DuckDuckGo search using the exact query
            2. After getting results, reflect on what information is missing
            3. If needed, perform additional targeted searches
            4. Only after gathering sufficient information, provide your final answer
            
            NEVER answer without searching first, even if you think you know the answer.
            """,
            
            AgentMode.ANALYSIS: f"""
            You are in ANALYSIS mode. For this query: "{query}"
            
            MANDATORY WORKFLOW:
            1. Break down the problem into components
            2. Use Math tool for any calculations needed
            3. Research any facts you're unsure about
            4. Synthesize findings with clear reasoning
            """,
            
            AgentMode.CREATIVE: f"""
            You are in CREATIVE mode. For this query: "{query}"
            
            WORKFLOW:
            1. Research current trends and examples related to the topic
            2. Generate multiple creative approaches
            3. Refine the best ideas based on research
            """,
            
            AgentMode.PROBLEM_SOLVING: f"""
            You are in PROBLEM_SOLVING mode. For this query: "{query}"
            
            MANDATORY WORKFLOW:
            1. Define the problem clearly
            2. Research similar problems and solutions
            3. Generate potential solutions
            4. Evaluate each solution
            5. Recommend the best approach
            """
        }
        return prompts.get(self.mode, prompts[AgentMode.RESEARCH])
    
    def invoke(self, inputs, config=None, **kwargs):
        """Override invoke to add control logic"""
        
        # Reset counters for new invocation
        self.iteration_count = 0
        self.reflection_history = []
        
        # Add mode-specific instructions to the input
        if isinstance(inputs, dict) and "input" in inputs:
            mode_prompt = self._get_mode_specific_prompt(inputs["input"])
            inputs["input"] = f"{mode_prompt}\n\nUser Query: {inputs['input']}"
        
        return super().invoke(inputs, config, **kwargs)

class ReflectiveAgent:
    """Agent that incorporates reflection stages"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.thought_history = []
    
    def _reflect_on_progress(self, query: str, actions_taken: List[str], results: List[str]) -> str:
        """Generate reflection on current progress"""
        
        reflection_prompt = f"""
        Original Query: {query}
        
        Actions taken so far:
        {chr(10).join(f"- {action}" for action in actions_taken)}
        
        Results obtained:
        {chr(10).join(f"- {result[:200]}..." for result in results)}
        
        Please reflect on:
        1. How well do the current results address the original query?
        2. What key information is still missing?
        3. What should be the next logical step?
        4. Are we on the right track or should we change approach?
        
        Provide a concise reflection (max 3 sentences):
        """
        
        response = self.llm.invoke([HumanMessage(content=reflection_prompt)])
        reflection = response.content
        
        self.thought_history.append({
            "type": "reflection",
            "content": reflection,
            "actions_taken": actions_taken.copy(),
            "results_count": len(results)
        })
        
        return reflection
    
    def _create_strategic_plan(self, query: str) -> str:
        """Create a strategic plan for approaching the query"""
        
        planning_prompt = f"""
        Query: {query}
        
        Create a strategic plan with:
        1. Initial assessment: What type of query is this?
        2. Information needs: What specific information do we need?
        3. Tool strategy: Which tools should we use and in what order?
        4. Success criteria: How will we know when we have enough information?
        
        Strategic Plan:
        """
        
        response = self.llm.invoke([HumanMessage(content=planning_prompt)])
        plan = response.content
        
        self.thought_history.append({
            "type": "strategic_plan",
            "content": plan
        })
        
        return plan
    
    def solve_with_reflection(self, query: str) -> str:
        """Solve query with built-in reflection stages"""
        
        # Stage 1: Strategic Planning
        plan = self._create_strategic_plan(query)
        print(f"📋 Strategic Plan:\n{plan}\n")
        
        actions_taken = []
        results = []
        
        # Stage 2: Initial Research
        search_tool = next((tool for tool in self.tools if tool.name == "DuckDuckGo"), None)
        if search_tool:
            try:
                initial_search = search_tool.func(query=query)
                actions_taken.append(f"Initial search for: {query}")
                results.append(initial_search)
                print(f"🔍 Initial Search Results:\n{initial_search[:300]}...\n")
            except Exception as e:
                print(f"Search failed: {e}")
        
        # Stage 3: First Reflection
        if actions_taken:
            reflection1 = self._reflect_on_progress(query, actions_taken, results)
            print(f"🤔 First Reflection:\n{reflection1}\n")
        
        # Stage 4: Additional Research (if needed)
        if len(results) < 2:  # Simple heuristic
            wiki_tool = next((tool for tool in self.tools if tool.name == "Wikipedia"), None)
            if wiki_tool:
                try:
                    wiki_search = wiki_tool.func(query=query)
                    actions_taken.append(f"Wikipedia search for: {query}")
                    results.append(wiki_search)
                    print(f"📚 Wikipedia Results:\n{wiki_search[:300]}...\n")
                except Exception as e:
                    print(f"Wikipedia search failed: {e}")
        
        # Stage 5: Final Reflection and Synthesis
        if len(actions_taken) > 1:
            final_reflection = self._reflect_on_progress(query, actions_taken, results)
            print(f"🎯 Final Reflection:\n{final_reflection}\n")
        
        # Stage 6: Synthesis
        synthesis_prompt = f"""
        Original Query: {query}
        
        Strategic Plan: {plan}
        
        Research Results:
        {chr(10).join(results)}
        
        Reflections:
        {chr(10).join([thought["content"] for thought in self.thought_history if thought["type"] == "reflection"])}
        
        Provide a comprehensive answer that:
        1. Directly addresses the original query
        2. Incorporates all relevant research findings
        3. Acknowledges any limitations or uncertainties
        
        Final Answer:
        """
        
        final_response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        return final_response.content

class RuleBasedAgent:
    """Agent with explicit rules and constraints"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], rules: Dict[str, Any]):
        self.llm = llm
        self.tools = tools
        self.rules = rules
    
    def _check_rules(self, query: str, action: str) -> bool:
        """Check if the proposed action follows the rules"""
        
        # Example rules
        if "research" in query.lower() and action != "search_first":
            if self.rules.get("research_must_search_first", True):
                return False
        
        if "calculate" in query.lower() and "Math" not in action:
            if self.rules.get("calculations_require_math_tool", True):
                return False
        
        return True
    
    def _get_mandatory_first_action(self, query: str) -> Optional[str]:
        """Get mandatory first action based on the query type and rules"""
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["research", "find", "search", "latest", "current"]):
            return "search"
        elif any(keyword in query_lower for keyword in ["calculate", "compute", "math", "solve"]):
            return "calculate"
        elif any(keyword in query_lower for keyword in ["time", "date", "when", "current time"]):
            return "get_time"
            
        return None
    
    def process_with_rules(self, query: str) -> str:
        """Process query while enforcing rules"""
        
        # Check mandatory first action
        mandatory_action = self._get_mandatory_first_action(query)
        
        if mandatory_action == "search":
            print("🔒 Rule: Research queries must start with search")
            search_tool = next((tool for tool in self.tools if tool.name == "DuckDuckGo"), None)
            if search_tool:
                search_result = search_tool.func(query=query)
                print(f"✅ Mandatory search completed: {search_result[:200]}...")
        
        elif mandatory_action == "calculate":
            print("🔒 Rule: Math queries must use Math tool")
            # Extract mathematical expression (simplified)
            math_tool = next((tool for tool in self.tools if tool.name == "Math"), None)
            if math_tool and any(op in query for op in ['+', '-', '*', '/', 'sqrt', 'pow']):
                try:
                    # This is a simplified extraction - in practice you'd need better parsing
                    math_result = math_tool.func(expression=query)
                    print(f"✅ Mandatory calculation completed: {math_result}")
                except:
                    print("⚠️ Could not extract mathematical expression")
        
        # Continue with normal processing...
        response = self.llm.invoke([HumanMessage(content=f"Query: {query}\nNote: Relevant rules have been enforced.")])
        return response.content

# Factory function to create different agent types
def create_controlled_agent(agent_type: str, llm: BaseLanguageModel, tools: List[BaseTool], **kwargs) -> Any:
    """Factory function to create different types of controlled agents"""
    
    if agent_type == "langgraph":
        from core.advanced_agent import create_advanced_agent
        return create_advanced_agent(llm, tools)
    
    elif agent_type == "reflective":
        return ReflectiveAgent(llm, tools)
    
    elif agent_type == "rule_based":
        rules = kwargs.get("rules", {
            "research_must_search_first": True,
            "calculations_require_math_tool": True,
            "max_iterations": 3
        })
        return RuleBasedAgent(llm, tools, rules)
    
    elif agent_type == "controlled_executor":
        mode = kwargs.get("mode", AgentMode.RESEARCH)
        
        # Create a basic agent structure
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Follow the mode-specific instructions carefully."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        llm_with_tools = llm.bind_tools(tools)
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x.get("intermediate_steps", [])
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        return ControlledAgentExecutor(
            agent=agent,
            tools=tools,
            mode=mode,
            verbose=True,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 