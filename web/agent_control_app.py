# web/agent_control_app.py
import streamlit as st
import asyncio
import json
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.autonomous_research_system import ResearchOrchestrator, AgentRole, TaskStatus

# Page config
st.set_page_config(
    page_title="🤖 Autonomous Research Command Center",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agent interface
st.markdown("""
<style>
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .research-mission {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .safety-high { border-left-color: #dc3545 !important; }
    .safety-medium { border-left-color: #ffc107 !important; }
    .safety-low { border-left-color: #28a745 !important; }
    .status-running { background-color: #e3f2fd; }
    .status-completed { background-color: #e8f5e8; }
    .status-failed { background-color: #ffebee; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = ResearchOrchestrator()
if 'active_missions' not in st.session_state:
    st.session_state.active_missions = {}
if 'mission_history' not in st.session_state:
    st.session_state.mission_history = []
if 'safety_settings' not in st.session_state:
    st.session_state.safety_settings = {
        'require_approval': True,
        'max_time_minutes': 30,
        'allowed_domains': ['google.com', 'wikipedia.org', 'github.com'],
        'blocked_actions': ['delete', 'purchase', 'download_executable']
    }

def main():
    st.title("🤖 Autonomous Research Command Center")
    st.markdown("**Deploy, monitor, and control your AI research agents**")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        page = st.selectbox(
            "Navigation",
            ["🚀 Mission Control", "👥 Agent Status", "📊 Analytics", "⚙️ Settings", "🛡️ Safety Monitor"]
        )
        
        st.divider()
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Missions", len(st.session_state.active_missions))
        with col2:
            st.metric("Completed Today", len([m for m in st.session_state.mission_history if m.get('date') == datetime.now().date()]))
        
        # Emergency stop
        st.divider()
        if st.button("🛑 EMERGENCY STOP ALL", type="primary"):
            st.session_state.active_missions.clear()
            st.success("All missions stopped!")
            st.rerun()
    
    # Main content based on selected page
    if page == "🚀 Mission Control":
        mission_control_page()
    elif page == "👥 Agent Status":
        agent_status_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "🛡️ Safety Monitor":
        safety_monitor_page()

def mission_control_page():
    st.header("🚀 Mission Control")
    
    # Mission launcher
    with st.expander("🎯 Launch New Research Mission", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            research_objective = st.text_area(
                "Research Objective",
                placeholder="e.g., 'Analyze the latest developments in quantum computing for business applications'",
                height=100
            )
        
        with col2:
            depth_level = st.selectbox("Depth Level", ["shallow", "medium", "deep"])
            time_limit = st.slider("Time Limit (minutes)", 5, 120, 30)
            require_approval = st.checkbox("Require Human Approval", value=True)
            
            # Advanced options
            with st.expander("Advanced Options"):
                search_domains = st.multiselect(
                    "Allowed Domains",
                    ["google.com", "wikipedia.org", "github.com", "arxiv.org", "reddit.com", "stackoverflow.com"],
                    default=["google.com", "wikipedia.org"]
                )
                
                use_vm = st.checkbox("Enable VM Code Execution", value=False)
                if use_vm:
                    st.warning("⚠️ VM execution allows code to run in sandboxed environment")
        
        # Launch button
        if st.button("🚀 Launch Research Mission", type="primary"):
            if research_objective.strip():
                mission_id = f"mission_{int(time.time())}"
                
                # Store mission in session state
                st.session_state.active_missions[mission_id] = {
                    "id": mission_id,
                    "objective": research_objective,
                    "depth": depth_level,
                    "time_limit": time_limit,
                    "status": "preparing",
                    "start_time": datetime.now(),
                    "progress": 0,
                    "require_approval": require_approval,
                    "allowed_domains": search_domains,
                    "use_vm": use_vm
                }
                
                st.success(f"✅ Mission {mission_id} queued for launch!")
                
                # Auto-run the mission (in a real app, this would be async)
                with st.spinner("🔄 Executing research mission..."):
                    try:
                        # This is a simulation - in reality you'd run this asynchronously
                        result = simulate_research_mission(mission_id, research_objective, depth_level, time_limit)
                        
                        # Update mission status
                        st.session_state.active_missions[mission_id].update({
                            "status": "completed",
                            "progress": 100,
                            "end_time": datetime.now(),
                            "result": result
                        })
                        
                        # Move to history
                        st.session_state.mission_history.append(st.session_state.active_missions[mission_id])
                        del st.session_state.active_missions[mission_id]
                        
                        st.success("🎉 Mission completed! Check results below.")
                        
                    except Exception as e:
                        st.error(f"❌ Mission failed: {str(e)}")
            else:
                st.error("Please enter a research objective")
    
    # Active missions
    st.subheader("⚡ Active Missions")
    if st.session_state.active_missions:
        for mission_id, mission in st.session_state.active_missions.items():
            display_mission_card(mission, active=True)
    else:
        st.info("No active missions. Launch one above!")
    
    # Recent missions
    st.subheader("📋 Recent Missions")
    if st.session_state.mission_history:
        # Show last 5 missions
        recent_missions = sorted(
            st.session_state.mission_history, 
            key=lambda x: x.get('start_time', datetime.min), 
            reverse=True
        )[:5]
        
        for mission in recent_missions:
            display_mission_card(mission, active=False)
    else:
        st.info("No completed missions yet.")

def display_mission_card(mission, active=True):
    """Display a mission card with status and controls"""
    
    status = mission.get('status', 'unknown')
    objective = mission.get('objective', 'Unknown objective')
    
    # Determine card class
    card_class = f"research-mission status-{status}"
    if mission.get('safety_level'):
        card_class += f" safety-{mission['safety_level']}"
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**Mission ID:** `{mission['id']}`")
            st.markdown(f"**Objective:** {objective}")
            
            if active:
                progress = mission.get('progress', 0)
                st.progress(progress / 100)
                st.caption(f"Status: {status.title()} • Progress: {progress}%")
            else:
                duration = mission.get('end_time', datetime.now()) - mission.get('start_time', datetime.now())
                st.caption(f"Completed in {duration.total_seconds()/60:.1f} minutes")
        
        with col2:
            depth = mission.get('depth', 'unknown')
            time_limit = mission.get('time_limit', 0)
            st.metric("Depth", depth.title())
            st.metric("Time Limit", f"{time_limit}m")
        
        with col3:
            if active:
                if st.button(f"⏹️ Stop", key=f"stop_{mission['id']}"):
                    st.session_state.active_missions.pop(mission['id'], None)
                    st.rerun()
            else:
                if st.button(f"📄 View Report", key=f"report_{mission['id']}"):
                    show_mission_report(mission)
        
        st.divider()

def show_mission_report(mission):
    """Show detailed mission report in modal"""
    
    with st.expander(f"📊 Mission Report: {mission['id']}", expanded=True):
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{((mission.get('end_time', datetime.now()) - mission.get('start_time', datetime.now())).total_seconds()/60):.1f}m")
        with col2:
            st.metric("Status", mission.get('status', 'Unknown').title())
        with col3:
            st.metric("Tasks", mission.get('tasks_completed', 0))
        
        # Results
        result = mission.get('result', {})
        if result:
            st.subheader("🔍 Research Findings")
            
            # Final report
            final_report = result.get('final_report', 'No report generated')
            st.markdown(final_report)
            
            # Detailed results
            if result.get('detailed_results'):
                with st.expander("📊 Detailed Results"):
                    for i, task_result in enumerate(result['detailed_results'], 1):
                        st.markdown(f"**Task {i}:** {task_result.get('task_id', 'Unknown')}")
                        if task_result.get('success'):
                            st.success("✅ Completed successfully")
                            if task_result.get('results', {}).get('summary'):
                                st.markdown(task_result['results']['summary'])
                        else:
                            st.error(f"❌ Failed: {task_result.get('error', 'Unknown error')}")
                        st.divider()

def agent_status_page():
    st.header("👥 Agent Status Dashboard")
    
    # Agent overview cards
    agents = [
        {"name": "Web Researcher", "status": "Ready", "tasks": 0, "success_rate": 95},
        {"name": "Data Analyst", "status": "Ready", "tasks": 0, "success_rate": 88},
        {"name": "Code Executor", "status": "Ready", "tasks": 0, "success_rate": 92},
        {"name": "Safety Monitor", "status": "Active", "tasks": 1, "success_rate": 100},
        {"name": "Report Writer", "status": "Ready", "tasks": 0, "success_rate": 97}
    ]
    
    cols = st.columns(len(agents))
    for i, agent in enumerate(agents):
        with cols[i]:
            status_color = "🟢" if agent["status"] == "Ready" else "🔵" if agent["status"] == "Active" else "🔴"
            st.markdown(f"""
            <div class="agent-card">
                <h4>{status_color} {agent['name']}</h4>
                <p>Status: {agent['status']}</p>
                <p>Active Tasks: {agent['tasks']}</p>
                <p>Success Rate: {agent['success_rate']}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Agent capabilities
    st.subheader("🛠️ Agent Capabilities")
    
    capabilities_data = {
        "Agent": ["Web Researcher", "Data Analyst", "Code Executor", "Safety Monitor", "Report Writer"],
        "Web Browsing": ["✅", "❌", "❌", "✅", "❌"],
        "Data Analysis": ["❌", "✅", "✅", "❌", "✅"],
        "Code Execution": ["❌", "✅", "✅", "❌", "❌"],
        "Safety Checks": ["✅", "❌", "❌", "✅", "❌"],
        "Report Generation": ["❌", "✅", "❌", "❌", "✅"]
    }
    
    df = pd.DataFrame(capabilities_data)
    st.dataframe(df, use_container_width=True)
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    # Generate sample performance data
    dates = pd.date_range(start='2025-08-01', end='2025-08-09', freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'Tasks Completed': [12, 15, 8, 22, 18, 25, 16, 20, 14],
        'Success Rate': [94, 96, 88, 98, 92, 99, 89, 95, 93],
        'Avg Duration (min)': [8.5, 7.2, 12.1, 6.8, 9.3, 5.9, 11.2, 7.8, 8.9]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(performance_data, x='Date', y='Tasks Completed', 
                      title='Tasks Completed Over Time')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(performance_data, x='Date', y='Success Rate', 
                      title='Success Rate Over Time')
        st.plotly_chart(fig2, use_container_width=True)

def analytics_page():
    st.header("📊 Research Analytics")
    
    # Generate sample analytics data
    if not st.session_state.mission_history:
        st.info("Complete some research missions to see analytics!")
        return
    
    # Mission statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_missions = len(st.session_state.mission_history)
    successful_missions = len([m for m in st.session_state.mission_history if m.get('status') == 'completed'])
    avg_duration = sum([(m.get('end_time', datetime.now()) - m.get('start_time', datetime.now())).total_seconds()/60 
                       for m in st.session_state.mission_history]) / total_missions if total_missions > 0 else 0
    
    with col1:
        st.metric("Total Missions", total_missions)
    with col2:
        st.metric("Success Rate", f"{(successful_missions/total_missions*100):.1f}%" if total_missions > 0 else "0%")
    with col3:
        st.metric("Avg Duration", f"{avg_duration:.1f}m")
    with col4:
        st.metric("Total Research Time", f"{sum([(m.get('end_time', datetime.now()) - m.get('start_time', datetime.now())).total_seconds()/60 for m in st.session_state.mission_history]):.1f}m")
    
    # Research topics analysis
    st.subheader("🎯 Research Topics")
    
    if st.session_state.mission_history:
        topics = [mission.get('objective', 'Unknown')[:50] + '...' if len(mission.get('objective', '')) > 50 
                 else mission.get('objective', 'Unknown') for mission in st.session_state.mission_history]
        
        topic_df = pd.DataFrame({'Topic': topics, 'Count': [1] * len(topics)})
        topic_counts = topic_df.groupby('Topic')['Count'].sum().reset_index()
        
        fig = px.bar(topic_counts, x='Topic', y='Count', title='Research Topics Frequency')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    st.subheader("📈 Performance Trends")
    
    # Time-based analysis
    if len(st.session_state.mission_history) > 1:
        history_df = pd.DataFrame(st.session_state.mission_history)
        history_df['start_date'] = pd.to_datetime([m.get('start_time', datetime.now()) for m in st.session_state.mission_history])
        history_df['duration'] = [(m.get('end_time', datetime.now()) - m.get('start_time', datetime.now())).total_seconds()/60 
                                 for m in st.session_state.mission_history]
        
        fig = px.scatter(history_df, x='start_date', y='duration', 
                        title='Mission Duration Over Time',
                        labels={'start_date': 'Date', 'duration': 'Duration (minutes)'})
        st.plotly_chart(fig, use_container_width=True)

def settings_page():
    st.header("⚙️ System Settings")
    
    # Safety settings
    st.subheader("🛡️ Safety Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        require_approval = st.checkbox(
            "Require Human Approval", 
            value=st.session_state.safety_settings['require_approval']
        )
        
        max_time = st.slider(
            "Maximum Mission Time (minutes)", 
            5, 180, 
            st.session_state.safety_settings['max_time_minutes']
        )
        
        # Allowed domains
        st.subheader("🌐 Allowed Domains")
        allowed_domains = st.text_area(
            "Domains (one per line)",
            value='\n'.join(st.session_state.safety_settings['allowed_domains']),
            height=100
        )
    
    with col2:
        # Blocked actions
        st.subheader("🚫 Blocked Actions")
        blocked_actions = st.text_area(
            "Actions (one per line)",
            value='\n'.join(st.session_state.safety_settings['blocked_actions']),
            height=100
        )
        
        # Resource limits
        st.subheader("💻 Resource Limits")
        max_memory = st.slider("Max VM Memory (GB)", 1, 8, 2)
        max_cpu_cores = st.slider("Max CPU Cores", 1, 4, 2)
        max_requests_per_hour = st.slider("Max Web Requests/Hour", 10, 1000, 100)
    
    # API configuration
    st.subheader("🔌 API Configuration")
    
    api_provider = st.selectbox(
        "Primary AI Provider",
        ["Together.ai", "OpenAI", "Anthropic", "Local Model"]
    )
    
    if api_provider == "Local Model":
        model_path = st.text_input("Model Path", placeholder="/path/to/model.gguf")
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.session_state.safety_settings.update({
            'require_approval': require_approval,
            'max_time_minutes': max_time,
            'allowed_domains': [d.strip() for d in allowed_domains.split('\n') if d.strip()],
            'blocked_actions': [a.strip() for a in blocked_actions.split('\n') if a.strip()]
        })
        st.success("✅ Settings saved successfully!")

def safety_monitor_page():
    st.header("🛡️ Safety Monitor")
    
    # Safety status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 Safe Actions", 1247, delta="+23")
    with col2:
        st.metric("🟡 Flagged Actions", 12, delta="+2")
    with col3:
        st.metric("🔴 Blocked Actions", 3, delta="0")
    with col4:
        st.metric("🛡️ Safety Score", "98.2%", delta="+0.1%")
    
    # Recent safety events
    st.subheader("📋 Recent Safety Events")
    
    safety_events = [
        {
            "timestamp": datetime.now() - timedelta(minutes=5),
            "event": "Attempted to access blocked domain",
            "action": "Blocked",
            "agent": "Web Researcher",
            "severity": "Medium"
        },
        {
            "timestamp": datetime.now() - timedelta(hours=2),
            "event": "High resource usage detected",
            "action": "Warning issued",
            "agent": "Code Executor", 
            "severity": "Low"
        },
        {
            "timestamp": datetime.now() - timedelta(hours=6),
            "event": "Attempted to execute restricted command",
            "action": "Blocked",
            "agent": "Code Executor",
            "severity": "High"
        }
    ]
    
    for event in safety_events:
        severity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[event["severity"]]
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.write(f"**{event['event']}**")
                st.caption(f"Agent: {event['agent']}")
            
            with col2:
                st.write(f"Action: {event['action']}")
                st.caption(event['timestamp'].strftime("%H:%M:%S"))
            
            with col3:
                st.write(f"{severity_color} {event['severity']}")
            
            with col4:
                if st.button("🔍", key=f"investigate_{event['timestamp']}"):
                    st.info("Investigation panel would open here")
            
            st.divider()
    
    # Safety rules configuration
    st.subheader("⚙️ Safety Rules")
    
    with st.expander("Configure Safety Rules"):
        rule_type = st.selectbox("Rule Type", ["Domain Restriction", "Action Block", "Resource Limit"])
        
        if rule_type == "Domain Restriction":
            domain = st.text_input("Domain to restrict")
            if st.button("Add Domain Restriction"):
                st.success(f"Added restriction for {domain}")
        
        elif rule_type == "Action Block":
            action = st.text_input("Action pattern to block")
            if st.button("Add Action Block"):
                st.success(f"Added block for action pattern: {action}")

def simulate_research_mission(mission_id: str, objective: str, depth: str, time_limit: int) -> dict:
    """Simulate a research mission for demo purposes"""
    
    # Simulate processing time
    time.sleep(2)
    
    # Generate fake results
    tasks_completed = 3 if depth == "shallow" else 5 if depth == "medium" else 8
    
    return {
        "session_id": mission_id,
        "objective": objective,
        "duration_minutes": min(time_limit * 0.7, time_limit),  # Usually finishes before time limit
        "tasks_completed": tasks_completed,
        "final_report": f"""
# Research Report: {objective}

## Executive Summary
Completed comprehensive research on the specified objective using autonomous AI agents.

## Key Findings
- {tasks_completed} research tasks completed successfully
- Multiple high-quality sources analyzed
- Data synthesis and pattern recognition performed
- Actionable insights generated

## Methodology
The research was conducted using a multi-agent system with the following approach:
1. **Planning Phase**: Objective decomposition and task prioritization
2. **Execution Phase**: Parallel web research and data collection
3. **Analysis Phase**: Data synthesis and insight extraction
4. **Reporting Phase**: Comprehensive report generation

## Conclusions
The research objective has been thoroughly investigated with {depth} level analysis.
Results indicate significant opportunities for further exploration in related areas.

## Next Steps
- Consider deeper analysis of specific findings
- Explore related research areas identified
- Implement findings in practical applications
        """,
        "detailed_results": [
            {
                "task_id": f"task_{i}",
                "success": True,
                "method": "web_search",
                "results": {
                    "summary": f"Task {i} completed successfully with relevant findings.",
                    "sources_found": 5
                }
            } for i in range(1, tasks_completed + 1)
        ]
    }

if __name__ == "__main__":
    main()