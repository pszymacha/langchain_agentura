"""
Advanced Agent with controlled workflow using LangGraph
"""

import operator
from typing import Annotated, Dict, Any, List, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    """State of the research agent"""
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], operator.add]
    query: str
    research_plan: str
    search_results: List[str]
    reflection: str
    next_action: str
    iteration_count: int
    final_answer: str

class AdvancedResearchAgent:
    """Advanced agent with controlled workflow for research tasks"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = True, recursion_limit: int = 50):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.checkpointer = MemorySaver()
        self.verbose = verbose
        self.recursion_limit = recursion_limit
        self.graph = self._build_graph()
    
    def _verbose_print(self, step: str, content: str, truncate_at: int = 300):
        """Print verbose output if enabled"""
        if self.verbose:
            print(f"\n🔄 {step}")
            print("─" * 50)
            if len(content) > truncate_at:
                print(f"{content[:truncate_at]}...")
                print(f"[Truncated at {truncate_at} characters]")
            else:
                print(content)
            print("─" * 50)
    
    def _build_graph(self) -> StateGraph:
        """Build the controlled workflow graph"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("create_research_plan", self._create_research_plan)
        workflow.add_node("execute_search", self._execute_search)
        workflow.add_node("reflect_on_results", self._reflect_on_results)
        workflow.add_node("decide_next_step", self._decide_next_step)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        
        # Define the flow
        workflow.add_edge(START, "classify_query")
        
        # Conditional edges based on query type
        workflow.add_conditional_edges(
            "classify_query",
            self._should_research,
            {
                "research": "create_research_plan",
                "direct": "synthesize_answer"
            }
        )
        
        workflow.add_edge("create_research_plan", "execute_search")
        workflow.add_edge("execute_search", "reflect_on_results")
        workflow.add_edge("reflect_on_results", "decide_next_step")
        
        # Conditional continuation
        workflow.add_conditional_edges(
            "decide_next_step",
            self._should_continue,
            {
                "continue": "execute_search",
                "finish": "synthesize_answer"
            }
        )
        
        workflow.add_edge("synthesize_answer", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _classify_query(self, state: AgentState) -> Dict[str, Any]:
        """Classify if query requires research or can be answered directly"""
        
        self._verbose_print("STEP 1: Query Classification", f"Analyzing query: {state['query']}")
        
        classification_prompt = f"""
        Analyze this query and determine if it requires internet research or can be answered directly:
        
        Query: {state['query']}
        
        Respond with either:
        - "RESEARCH" if it requires current information, facts, or external data
        - "DIRECT" if it can be answered with general knowledge
        
        Classification:"""
        
        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        classification = "research" if "RESEARCH" in response.content.upper() else "direct"
        
        self._verbose_print("Classification Result", f"Query classified as: {classification.upper()}")
        
        return {
            "messages": [AIMessage(content=f"Query classified as: {classification}")],
            "next_action": classification
        }
    
    def _should_research(self, state: AgentState) -> Literal["research", "direct"]:
        """Determine if research is needed"""
        return state.get("next_action", "direct")
    
    def _create_research_plan(self, state: AgentState) -> Dict[str, Any]:
        """Create a structured research plan"""
        
        self._verbose_print("STEP 2: Research Planning", "Creating strategic research plan...")
        
        planning_prompt = f"""
        Create a detailed research plan for this query: {state['query']}
        
        Your plan should include:
        1. Key search terms and phrases
        2. Types of information to look for
        3. Search strategy (broad first, then specific)
        4. Quality criteria for sources
        
        Research Plan:"""
        
        response = self.llm.invoke([HumanMessage(content=planning_prompt)])
        
        self._verbose_print("Research Plan Created", response.content)
        
        return {
            "messages": [AIMessage(content="Research plan created")],
            "research_plan": response.content,
            "iteration_count": 0
        }
    
    def _execute_search(self, state: AgentState) -> Dict[str, Any]:
        """Execute search based on current plan and iteration"""
        
        search_tool = self.tools.get("DuckDuckGo")
        if not search_tool:
            self._verbose_print("Search Error", "No search tool available")
            return {"messages": [AIMessage(content="No search tool available")]}
        
        # Determine search query based on iteration
        if state["iteration_count"] == 0:
            # First search - broad
            search_query = state["query"]
            self._verbose_print(f"STEP 3: Initial Search (Iteration {state['iteration_count'] + 1})", 
                              f"Executing broad search: {search_query}")
        else:
            # Subsequent searches - refined based on reflection
            search_query = f"{state['query']} {state.get('reflection', '')}"
            self._verbose_print(f"STEP 3: Refined Search (Iteration {state['iteration_count'] + 1})", 
                              f"Executing refined search: {search_query}")
        
        try:
            search_results = search_tool.func(query=search_query)
            
            self._verbose_print("Search Results", search_results)
            
            return {
                "messages": [AIMessage(content=f"Executed search: {search_query}")],
                "search_results": state.get("search_results", []) + [search_results],
                "iteration_count": state["iteration_count"] + 1
            }
        except Exception as e:
            self._verbose_print("Search Error", f"Search failed: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Search failed: {str(e)}")],
                "search_results": state.get("search_results", [])
            }
    
    def _reflect_on_results(self, state: AgentState) -> Dict[str, Any]:
        """Reflect on search results and plan next steps"""
        
        self._verbose_print(f"STEP 4: Reflection (After Iteration {state['iteration_count']})", 
                          "Analyzing search results and planning next steps...")
        
        reflection_prompt = f"""
        Original query: {state['query']}
        Research plan: {state['research_plan']}
        Current iteration: {state['iteration_count']}
        
        Latest search results:
        {state['search_results'][-1] if state['search_results'] else 'No results'}
        
        Reflect on:
        1. How well do current results answer the original query?
        2. What information is still missing?
        3. What should be the next search strategy?
        4. Should we continue searching or synthesize an answer?
        
        Reflection:"""
        
        response = self.llm.invoke([HumanMessage(content=reflection_prompt)])
        
        self._verbose_print("Reflection Complete", response.content)
        
        return {
            "messages": [AIMessage(content="Reflected on search results")],
            "reflection": response.content
        }
    
    def _decide_next_step(self, state: AgentState) -> Dict[str, Any]:
        """Decide whether to continue searching or finish"""
        
        self._verbose_print("STEP 5: Decision Making", "Deciding whether to continue or finish...")
        
        decision_prompt = f"""
        Based on the reflection: {state['reflection']}
        Current iteration: {state['iteration_count']}
        
        Should we:
        - CONTINUE searching (if more information needed, max 3 iterations)
        - FINISH and synthesize answer (if sufficient information gathered)
        
        Decision:"""
        
        response = self.llm.invoke([HumanMessage(content=decision_prompt)])
        
        # Determine next action
        should_continue = (
            "CONTINUE" in response.content.upper() and 
            state["iteration_count"] < 3
        )
        
        next_action = "continue" if should_continue else "finish"
        
        self._verbose_print("Decision Made", f"Decision: {next_action.upper()}")
        if next_action == "continue":
            self._verbose_print("", f"Will perform iteration {state['iteration_count'] + 1}")
        else:
            self._verbose_print("", "Moving to synthesis phase")
        
        return {
            "messages": [AIMessage(content=f"Decision: {next_action}")],
            "next_action": next_action
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "finish"]:
        """Determine if should continue or finish"""
        return state.get("next_action", "finish")
    
    def _synthesize_answer(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize final answer from all gathered information"""
        
        self._verbose_print("STEP 6: Final Synthesis", "Combining all research into comprehensive answer...")
        
        synthesis_prompt = f"""
        Original query: {state['query']}
        
        Research conducted:
        {state.get('research_plan', 'No research plan')}
        
        Search results:
        {chr(10).join(state.get('search_results', []))}
        
        Reflection:
        {state.get('reflection', 'No reflection')}
        
        Provide a comprehensive, well-structured answer to the original query based on all the research gathered.
        Include:
        1. Direct answer to the question
        2. Supporting evidence from research
        3. Any important caveats or limitations
        
        Final Answer:"""
        
        response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        
        self._verbose_print("Synthesis Complete", "Final answer generated successfully!")
        
        return {
            "messages": [AIMessage(content=response.content)],
            "final_answer": response.content
        }
    
    def research(self, query: str, thread_id: str = "default") -> str:
        """Execute the research workflow"""
        
        if self.verbose:
            print(f"\n🎯 Starting Advanced Research Workflow")
            print(f"📝 Query: {query}")
            print(f"🧵 Thread: {thread_id}")
            print("=" * 60)
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "research_plan": "",
            "search_results": [],
            "reflection": "",
            "next_action": "",
            "iteration_count": 0,
            "final_answer": ""
        }
        
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.recursion_limit
        }
        
        # Execute the workflow
        result = self.graph.invoke(initial_state, config)
        
        return result.get("final_answer", "No answer generated")

def create_advanced_agent(llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = True, recursion_limit: int = 50) -> AdvancedResearchAgent:
    """Create an advanced research agent with controlled workflow"""
    return AdvancedResearchAgent(llm, tools, verbose, recursion_limit) 