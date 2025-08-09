# src/agents/autonomous_research_system.py
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import os
import docker
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pyautogui
import requests
from pathlib import Path

# Import your existing AI playground
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.working_ai_playground import AIPlayground

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    WEB_RESEARCHER = "web_researcher"
    DATA_ANALYST = "data_analyst"
    CODE_EXECUTOR = "code_executor"
    SAFETY_MONITOR = "safety_monitor"
    REPORT_WRITER = "report_writer"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class ResearchTask:
    id: str
    description: str
    assigned_agent: AgentRole
    status: TaskStatus
    priority: int
    dependencies: List[str]
    results: Dict[str, Any]
    safety_level: str  # "low", "medium", "high"
    created_at: datetime
    updated_at: datetime

class SafetyGuards:
    """Comprehensive safety system for autonomous agents"""
    
    def __init__(self):
        self.blocked_domains = [
            "facebook.com", "twitter.com", "instagram.com",  # Social media
            "4chan.org", "8kun.top",  # Potentially harmful sites
            # Add more based on your needs
        ]
        self.blocked_actions = [
            "delete", "rm -rf", "format", "shutdown",
            "purchase", "buy", "order", "payment"
        ]
        self.max_budget_per_session = 0.0  # No spending without approval
        self.require_approval_for = [
            "financial_actions", "system_changes", "data_deletion"
        ]
    
    def check_url_safety(self, url: str) -> bool:
        """Check if URL is safe to visit"""
        for blocked in self.blocked_domains:
            if blocked in url.lower():
                return False
        return True
    
    def check_action_safety(self, action: str) -> bool:
        """Check if action is safe to execute"""
        action_lower = action.lower()
        for blocked in self.blocked_actions:
            if blocked in action_lower:
                return False
        return True
    
    def requires_human_approval(self, task: ResearchTask) -> bool:
        """Determine if task requires human approval"""
        return (
            task.safety_level == "high" or
            any(req in task.description.lower() for req in self.require_approval_for)
        )

class WebNavigatorAgent:
    """Agent capable of browsing the web autonomously"""
    
    def __init__(self, headless: bool = True):
        self.driver = None
        self.headless = headless
        self.safety_guards = SafetyGuards()
        self.session_log = []
        
    def start_browser(self):
        """Initialize the web browser"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Add user agent to appear more human
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        self.driver = webdriver.Chrome(options=chrome_options)
        return self.driver
    
    def navigate_to_url(self, url: str) -> Dict[str, Any]:
        """Safely navigate to a URL"""
        if not self.safety_guards.check_url_safety(url):
            return {"success": False, "error": f"URL blocked by safety guards: {url}"}
        
        try:
            if not self.driver:
                self.start_browser()
            
            self.driver.get(url)
            time.sleep(2)  # Allow page to load
            
            result = {
                "success": True,
                "url": self.driver.current_url,
                "title": self.driver.title,
                "page_source_length": len(self.driver.page_source),
                "timestamp": datetime.now().isoformat()
            }
            
            self.session_log.append(f"Navigated to: {url}")
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_text_content(self) -> str:
        """Extract clean text from current page"""
        if not self.driver:
            return ""
        
        try:
            # Remove script and style elements
            self.driver.execute_script("""
                var scripts = document.getElementsByTagName('script');
                for(var i = scripts.length - 1; i >= 0; i--) {
                    scripts[i].parentNode.removeChild(scripts[i]);
                }
                var styles = document.getElementsByTagName('style');
                for(var i = styles.length - 1; i >= 0; i--) {
                    styles[i].parentNode.removeChild(styles[i]);
                }
            """)
            
            # Get clean text
            body = self.driver.find_element(By.TAG_NAME, "body")
            return body.text
            
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def search_google(self, query: str) -> List[Dict[str, str]]:
        """Perform Google search and extract results"""
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        nav_result = self.navigate_to_url(search_url)
        if not nav_result["success"]:
            return []
        
        try:
            results = []
            search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            
            for result in search_results[:10]:  # Top 10 results
                try:
                    title_elem = result.find_element(By.CSS_SELECTOR, "h3")
                    link_elem = result.find_element(By.CSS_SELECTOR, "a")
                    snippet_elem = result.find_element(By.CSS_SELECTOR, "span")
                    
                    results.append({
                        "title": title_elem.text,
                        "url": link_elem.get_attribute("href"),
                        "snippet": snippet_elem.text
                    })
                except:
                    continue
            
            return results
            
        except Exception as e:
            return []
    
    def close_browser(self):
        """Clean up browser resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None

class VMControlAgent:
    """Agent for controlling virtual machines and computer interfaces"""
    
    def __init__(self, vm_type: str = "docker"):
        self.vm_type = vm_type
        self.docker_client = None
        self.current_container = None
        self.safety_guards = SafetyGuards()
        
    def start_research_vm(self) -> Dict[str, Any]:
        """Start a sandboxed Ubuntu VM for research"""
        try:
            if self.vm_type == "docker":
                self.docker_client = docker.from_env()
                
                # Create Ubuntu container with research tools
                self.current_container = self.docker_client.containers.run(
                    "ubuntu:22.04",
                    command="sleep infinity",
                    detach=True,
                    mem_limit="2g",
                    cpuset_cpus="0-1",  # Limit CPU usage
                    network_mode="bridge",
                    volumes={
                        "/tmp/research_data": {"bind": "/workspace", "mode": "rw"}
                    }
                )
                
                # Install basic research tools
                self.execute_in_vm([
                    "apt-get update",
                    "apt-get install -y python3 python3-pip curl wget git nano",
                    "pip3 install requests beautifulsoup4 pandas numpy"
                ])
                
                return {
                    "success": True,
                    "container_id": self.current_container.id,
                    "status": "running"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_in_vm(self, commands: List[str]) -> Dict[str, Any]:
        """Execute commands safely in VM"""
        if not self.current_container:
            return {"success": False, "error": "No VM running"}
        
        results = []
        for command in commands:
            # Safety check
            if not self.safety_guards.check_action_safety(command):
                results.append({
                    "command": command,
                    "success": False,
                    "error": "Command blocked by safety guards"
                })
                continue
            
            try:
                exec_result = self.current_container.exec_run(
                    f"bash -c '{command}'",
                    detach=False
                )
                
                results.append({
                    "command": command,
                    "success": exec_result.exit_code == 0,
                    "output": exec_result.output.decode("utf-8"),
                    "exit_code": exec_result.exit_code
                })
                
            except Exception as e:
                results.append({
                    "command": command,
                    "success": False,
                    "error": str(e)
                })
        
        return {"success": True, "results": results}
    
    def stop_vm(self):
        """Clean up VM resources"""
        if self.current_container:
            self.current_container.stop()
            self.current_container.remove()
            self.current_container = None

class ResearchOrchestrator:
    """Master orchestrator for autonomous research missions"""
    
    def __init__(self):
        self.ai_playground = AIPlayground()
        self.web_agent = WebNavigatorAgent(headless=True)
        self.vm_agent = VMControlAgent()
        self.safety_guards = SafetyGuards()
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.research_session_log = []
        self.human_approval_queue = []
        
    async def conduct_autonomous_research(
        self, 
        research_objective: str,
        depth_level: str = "medium",  # "shallow", "medium", "deep"
        time_limit_minutes: int = 30,
        require_human_approval: bool = True
    ) -> Dict[str, Any]:
        """
        Conduct autonomous research on a given objective
        
        Args:
            research_objective: What to research
            depth_level: How deep to go
            time_limit_minutes: Maximum time to spend
            require_human_approval: Whether to ask for approval on risky tasks
        """
        
        session_id = f"research_{int(time.time())}"
        start_time = datetime.now()
        
        print(f"🔬 Starting autonomous research session: {session_id}")
        print(f"🎯 Objective: {research_objective}")
        print(f"📊 Depth Level: {depth_level}")
        print(f"⏱️ Time Limit: {time_limit_minutes} minutes")
        
        # Phase 1: Planning and task decomposition
        research_plan = await self._create_research_plan(research_objective, depth_level)
        
        if require_human_approval:
            print(f"\n📋 Research Plan Created:")
            for i, task in enumerate(research_plan["tasks"], 1):
                print(f"   {i}. {task['description']} (Risk: {task['safety_level']})")
            
            approval = input("\n🤔 Approve this research plan? (y/n): ").lower().strip()
            if approval != 'y':
                return {"success": False, "message": "Research plan not approved by human"}
        
        # Phase 2: Execute research plan
        results = await self._execute_research_plan(research_plan, time_limit_minutes)
        
        # Phase 3: Synthesize findings
        final_report = await self._synthesize_research_findings(
            research_objective, 
            results
        )
        
        # Cleanup
        self.web_agent.close_browser()
        self.vm_agent.stop_vm()
        
        return {
            "success": True,
            "session_id": session_id,
            "objective": research_objective,
            "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
            "tasks_completed": len([r for r in results if r["success"]]),
            "final_report": final_report,
            "detailed_results": results
        }
    
    async def _create_research_plan(self, objective: str, depth: str) -> Dict[str, Any]:
        """Use AI to create a detailed research plan"""
        
        planning_prompt = f"""
        Create a detailed research plan for: "{objective}"
        
        Depth level: {depth}
        Available capabilities:
        - Web browsing and search
        - Data extraction and analysis
        - Code execution in sandboxed environment
        - Document analysis
        
        Create 3-7 specific tasks that would thoroughly research this topic.
        For each task, specify:
        1. Description (what to do)
        2. Method (web search, data analysis, etc.)
        3. Expected outcome
        4. Safety level (low/medium/high)
        5. Dependencies (which tasks must complete first)
        
        Format as JSON with this structure:
        {{
            "objective": "{objective}",
            "estimated_time_minutes": 20,
            "tasks": [
                {{
                    "id": "task_1",
                    "description": "Search for recent developments in...",
                    "method": "web_search",
                    "expected_outcome": "List of key findings",
                    "safety_level": "low",
                    "dependencies": []
                }}
            ]
        }}
        """
        
        plan_response = self.ai_playground.business_advisor(planning_prompt)
        
        try:
            # Extract JSON from AI response
            import re
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                return plan_data
        except:
            pass
        
        # Fallback plan if AI parsing fails
        return {
            "objective": objective,
            "estimated_time_minutes": 15,
            "tasks": [
                {
                    "id": "task_1",
                    "description": f"Search for information about {objective}",
                    "method": "web_search",
                    "expected_outcome": "Initial findings",
                    "safety_level": "low",
                    "dependencies": []
                },
                {
                    "id": "task_2", 
                    "description": f"Analyze and synthesize findings about {objective}",
                    "method": "analysis",
                    "expected_outcome": "Summary report",
                    "safety_level": "low",
                    "dependencies": ["task_1"]
                }
            ]
        }
    
    async def _execute_research_plan(
        self, 
        plan: Dict[str, Any], 
        time_limit: int
    ) -> List[Dict[str, Any]]:
        """Execute the research plan with all agents"""
        
        results = []
        start_time = time.time()
        
        for task_data in plan["tasks"]:
            # Check time limit
            if (time.time() - start_time) / 60 > time_limit:
                print(f"⏰ Time limit reached, stopping research")
                break
            
            task = ResearchTask(
                id=task_data["id"],
                description=task_data["description"],
                assigned_agent=AgentRole.WEB_RESEARCHER,  # Default assignment
                status=TaskStatus.PENDING,
                priority=1,
                dependencies=task_data.get("dependencies", []),
                results={},
                safety_level=task_data.get("safety_level", "medium"),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            print(f"\n🔄 Executing: {task.description}")
            
            # Check if dependencies are met
            if not self._dependencies_met(task, results):
                print(f"⏸️ Dependencies not met, skipping task")
                continue
            
            # Execute based on method
            method = task_data.get("method", "web_search")
            task_result = await self._execute_single_task(task, method)
            
            results.append(task_result)
            
            if task_result["success"]:
                print(f"✅ Task completed successfully")
            else:
                print(f"❌ Task failed: {task_result.get('error', 'Unknown error')}")
        
        return results
    
    async def _execute_single_task(
        self, 
        task: ResearchTask, 
        method: str
    ) -> Dict[str, Any]:
        """Execute a single research task"""
        
        try:
            if method == "web_search":
                return await self._web_search_task(task)
            elif method == "data_analysis":
                return await self._data_analysis_task(task)
            elif method == "code_execution":
                return await self._code_execution_task(task)
            else:
                return await self._web_search_task(task)  # Default
                
        except Exception as e:
            return {
                "task_id": task.id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _web_search_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute web search and analysis"""
        
        # Extract search query from task description using AI
        query_prompt = f"Extract 1-3 good Google search queries for: {task.description}"
        query_response = self.ai_playground.chat(query_prompt)
        
        # Simple query extraction (could be improved)
        search_queries = [q.strip() for q in query_response.split('\n') if q.strip()][:3]
        
        all_results = []
        
        for query in search_queries:
            print(f"   🔍 Searching: {query}")
            search_results = self.web_agent.search_google(query)
            
            # Visit top results and extract content
            for result in search_results[:3]:  # Top 3 per query
                if self.safety_guards.check_url_safety(result["url"]):
                    nav_result = self.web_agent.navigate_to_url(result["url"])
                    if nav_result["success"]:
                        content = self.web_agent.extract_text_content()
                        all_results.append({
                            "url": result["url"],
                            "title": result["title"],
                            "content": content[:2000],  # Limit content length
                            "source": "web_search"
                        })
        
        # Summarize findings with AI
        summary_prompt = f"""
        Research task: {task.description}
        
        Found {len(all_results)} web sources. Summarize key findings:
        
        {json.dumps(all_results, indent=2)}
        
        Provide a concise summary of the most important information found.
        """
        
        summary = self.ai_playground.business_advisor(summary_prompt)
        
        return {
            "task_id": task.id,
            "success": True,
            "method": "web_search",
            "results": {
                "search_queries": search_queries,
                "sources_found": len(all_results),
                "raw_data": all_results,
                "summary": summary
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _data_analysis_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute data analysis tasks"""
        # This would analyze data from previous tasks
        return {
            "task_id": task.id,
            "success": True,
            "method": "data_analysis",
            "results": {"analysis": "Data analysis not yet implemented"},
            "timestamp": datetime.now().isoformat()
        }
    
    async def _code_execution_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute code in sandboxed environment"""
        # Start VM if not running
        if not self.vm_agent.current_container:
            vm_result = self.vm_agent.start_research_vm()
            if not vm_result["success"]:
                return {
                    "task_id": task.id,
                    "success": False,
                    "error": "Could not start VM",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Generate code using AI
        code_prompt = f"""
        Create Python code to help with this research task: {task.description}
        
        The code will run in an Ubuntu container with basic Python libraries.
        Focus on data gathering, analysis, or processing that would help research this topic.
        
        Provide only the Python code, no explanations.
        """
        
        code = self.ai_playground.code_assistant(code_prompt)
        
        # Execute code in VM
        exec_result = self.vm_agent.execute_in_vm([f"python3 -c \"{code}\""])
        
        return {
            "task_id": task.id,
            "success": exec_result["success"],
            "method": "code_execution",
            "results": {
                "generated_code": code,
                "execution_result": exec_result
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _dependencies_met(self, task: ResearchTask, completed_results: List[Dict]) -> bool:
        """Check if task dependencies are satisfied"""
        completed_task_ids = [r["task_id"] for r in completed_results if r["success"]]
        return all(dep in completed_task_ids for dep in task.dependencies)
    
    async def _synthesize_research_findings(
        self, 
        objective: str, 
        results: List[Dict[str, Any]]
    ) -> str:
        """Create final research report"""
        
        findings_summary = []
        for result in results:
            if result["success"]:
                findings_summary.append(f"Task: {result['task_id']}")
                findings_summary.append(f"Results: {result['results']}")
                findings_summary.append("---")
        
        synthesis_prompt = f"""
        Research Objective: {objective}
        
        Findings from autonomous research:
        {chr(10).join(findings_summary)}
        
        Create a comprehensive research report that:
        1. Summarizes key findings
        2. Identifies patterns and insights
        3. Draws conclusions
        4. Suggests next steps or areas for deeper research
        
        Make it professional and actionable.
        """
        
        final_report = self.ai_playground.business_advisor(synthesis_prompt)
        return final_report

# Usage example and CLI interface
if __name__ == "__main__":
    async def main():
        orchestrator = ResearchOrchestrator()
        
        # Example research mission
        result = await orchestrator.conduct_autonomous_research(
            research_objective="AI agent market trends and competitive landscape 2025",
            depth_level="medium",
            time_limit_minutes=15,
            require_human_approval=True
        )
        
        print("\n" + "="*60)
        print("🎯 AUTONOMOUS RESEARCH COMPLETE")
        print("="*60)
        print(f"Session ID: {result['session_id']}")
        print(f"Duration: {result['duration_minutes']:.1f} minutes")
        print(f"Tasks Completed: {result['tasks_completed']}")
        print("\n📋 FINAL REPORT:")
        print("-"*40)
        print(result['final_report'])
        
    # Run the research mission
    asyncio.run(main())