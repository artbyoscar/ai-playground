# src/agents/safe_computer_control.py
"""
Safe Computer Control with Human Supervision
This is how we ACTUALLY implement safe AI agents
"""

import pyautogui
import psutil
import subprocess
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time
from pathlib import Path

class PermissionLevel(Enum):
    """Safety levels for operations"""
    READ_ONLY = "read_only"       # Can only observe
    SUPERVISED = "supervised"       # Requires confirmation
    SANDBOXED = "sandboxed"        # Limited to sandbox
    TRUSTED = "trusted"            # Full access (dangerous!)

class ActionType(Enum):
    """Types of actions the agent can take"""
    # Safe actions
    SCREENSHOT = "screenshot"
    READ_FILE = "read_file"
    SYSTEM_INFO = "system_info"
    
    # Medium risk
    OPEN_APP = "open_app"
    TYPE_TEXT = "type_text"
    CLICK = "click"
    
    # High risk
    DELETE_FILE = "delete_file"
    INSTALL_SOFTWARE = "install"
    MODIFY_SYSTEM = "modify_system"

@dataclass
class ActionRequest:
    """Request for an action with safety info"""
    action: ActionType
    description: str
    risk_level: str
    parameters: Dict[str, Any]
    reason: str

class SafeComputerAgent:
    """
    Computer control with safety guardrails
    """
    
    def __init__(self, permission_level: PermissionLevel = PermissionLevel.SUPERVISED):
        self.permission_level = permission_level
        self.action_log = []
        self.sandbox_dir = Path("./sandbox")
        self.sandbox_dir.mkdir(exist_ok=True)
        
        # Define risk levels for actions
        self.action_risks = {
            ActionType.SCREENSHOT: "low",
            ActionType.READ_FILE: "low",
            ActionType.SYSTEM_INFO: "low",
            ActionType.OPEN_APP: "medium",
            ActionType.TYPE_TEXT: "medium",
            ActionType.CLICK: "medium",
            ActionType.DELETE_FILE: "high",
            ActionType.INSTALL_SOFTWARE: "critical",
            ActionType.MODIFY_SYSTEM: "critical"
        }
        
        # Whitelisted applications (safe to open)
        self.safe_apps = [
            "notepad.exe",
            "calculator.exe",
            "mspaint.exe",
            "chrome.exe",
            "firefox.exe"
        ]
        
        # Blacklisted paths (never allow access)
        self.protected_paths = [
            "C:\\Windows\\System32",
            "C:\\Program Files",
            "~/Documents/passwords",
            "~/.ssh"
        ]
        
        print(f"🛡️ Safe Computer Agent initialized")
        print(f"📊 Permission Level: {permission_level.value}")
        print(f"📁 Sandbox: {self.sandbox_dir.absolute()}")
    
    def request_permission(self, action: ActionRequest) -> bool:
        """Ask human for permission"""
        print("\n" + "="*50)
        print("🤖 AI AGENT REQUEST")
        print("="*50)
        print(f"Action: {action.action.value}")
        print(f"Description: {action.description}")
        print(f"Risk Level: {action.risk_level}")
        print(f"Reason: {action.reason}")
        print(f"Parameters: {json.dumps(action.parameters, indent=2)}")
        print("="*50)
        
        if self.permission_level == PermissionLevel.READ_ONLY:
            print("❌ Denied: Read-only mode")
            return False
        
        if action.risk_level == "critical":
            print("⚠️ CRITICAL ACTION - Are you SURE?")
            confirm = input("Type 'YES I AM SURE' to allow: ")
            return confirm == "YES I AM SURE"
        
        if self.permission_level == PermissionLevel.SUPERVISED:
            response = input("Allow this action? (y/n): ")
            return response.lower() == 'y'
        
        if self.permission_level == PermissionLevel.SANDBOXED:
            # Auto-approve if within sandbox
            if self._is_sandboxed(action):
                print("✅ Auto-approved (sandboxed)")
                return True
            else:
                print("❌ Denied: Outside sandbox")
                return False
        
        return True
    
    def _is_sandboxed(self, action: ActionRequest) -> bool:
        """Check if action is within sandbox"""
        if action.action in [ActionType.READ_FILE, ActionType.DELETE_FILE]:
            path = Path(action.parameters.get("path", ""))
            return self.sandbox_dir in path.parents
        
        if action.action == ActionType.OPEN_APP:
            app = action.parameters.get("app", "")
            return app in self.safe_apps
        
        return action.risk_level == "low"
    
    def execute_action(self, action: ActionRequest) -> Any:
        """Execute an action with safety checks"""
        
        # Log the action
        self.action_log.append({
            "timestamp": time.time(),
            "action": action.action.value,
            "parameters": action.parameters,
            "approved": False
        })
        
        # Check permission
        if not self.request_permission(action):
            print("❌ Action denied by safety system")
            return None
        
        # Mark as approved
        self.action_log[-1]["approved"] = True
        
        # Execute based on type
        if action.action == ActionType.SCREENSHOT:
            return self._safe_screenshot()
        
        elif action.action == ActionType.READ_FILE:
            return self._safe_read_file(action.parameters["path"])
        
        elif action.action == ActionType.OPEN_APP:
            return self._safe_open_app(action.parameters["app"])
        
        elif action.action == ActionType.TYPE_TEXT:
            return self._safe_type_text(action.parameters["text"])
        
        elif action.action == ActionType.CLICK:
            return self._safe_click(
                action.parameters["x"],
                action.parameters["y"]
            )
        
        else:
            print(f"⚠️ Action {action.action} not implemented")
            return None
    
    def _safe_screenshot(self):
        """Take screenshot safely"""
        screenshot = pyautogui.screenshot()
        path = self.sandbox_dir / f"screenshot_{int(time.time())}.png"
        screenshot.save(path)
        print(f"✅ Screenshot saved: {path}")
        return path
    
    def _safe_read_file(self, path: str):
        """Read file with safety checks"""
        path = Path(path)
        
        # Check if path is protected
        for protected in self.protected_paths:
            if str(path).startswith(protected):
                print(f"❌ Access denied: Protected path")
                return None
        
        if not path.exists():
            print(f"❌ File not found: {path}")
            return None
        
        # Size limit (10MB)
        if path.stat().st_size > 10 * 1024 * 1024:
            print(f"❌ File too large (>10MB)")
            return None
        
        content = path.read_text()
        print(f"✅ Read {len(content)} characters from {path}")
        return content
    
    def _safe_open_app(self, app: str):
        """Open application safely"""
        if app not in self.safe_apps:
            print(f"⚠️ Warning: {app} not in whitelist")
        
        try:
            subprocess.Popen(app)
            print(f"✅ Opened {app}")
            return True
        except Exception as e:
            print(f"❌ Failed to open {app}: {e}")
            return False
    
    def _safe_type_text(self, text: str):
        """Type text with safety limits"""
        # Limit text length
        if len(text) > 1000:
            text = text[:1000]
            print("⚠️ Text truncated to 1000 chars")
        
        # Remove potentially dangerous characters
        dangerous = ['`', '$', '\\', '|', '>', '<', '&']
        for char in dangerous:
            if char in text:
                print(f"⚠️ Removed dangerous character: {char}")
                text = text.replace(char, '')
        
        pyautogui.typewrite(text, interval=0.05)
        print(f"✅ Typed {len(text)} characters")
        return True
    
    def _safe_click(self, x: int, y: int):
        """Click with boundary checks"""
        screen_width, screen_height = pyautogui.size()
        
        # Boundary check
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        
        # Warn about system areas
        if y < 50:  # Top menu bar area
            print("⚠️ Warning: Clicking near system menu")
        
        pyautogui.click(x, y)
        print(f"✅ Clicked at ({x}, {y})")
        return True
    
    def demonstrate_safe_workflow(self):
        """Demo of safe computer control"""
        print("\n" + "="*60)
        print("🛡️ SAFE COMPUTER CONTROL DEMONSTRATION")
        print("="*60)
        
        # Safe action - always allowed
        action1 = ActionRequest(
            action=ActionType.SCREENSHOT,
            description="Take a screenshot",
            risk_level="low",
            parameters={},
            reason="User requested system status check"
        )
        
        print("\n1️⃣ Safe Action (Screenshot):")
        self.execute_action(action1)
        
        # Medium risk - requires confirmation
        action2 = ActionRequest(
            action=ActionType.OPEN_APP,
            description="Open Notepad",
            risk_level="medium",
            parameters={"app": "notepad.exe"},
            reason="Need to create a text file for user"
        )
        
        print("\n2️⃣ Medium Risk (Open App):")
        self.execute_action(action2)
        
        # Sandboxed action
        sandbox_file = self.sandbox_dir / "test.txt"
        sandbox_file.write_text("Test content")
        
        action3 = ActionRequest(
            action=ActionType.READ_FILE,
            description="Read sandbox file",
            risk_level="low",
            parameters={"path": str(sandbox_file)},
            reason="Reading user-created test file"
        )
        
        print("\n3️⃣ Sandboxed Action (Read File):")
        self.execute_action(action3)
        
        # Show action log
        print("\n📊 Action Log:")
        for entry in self.action_log:
            status = "✅" if entry["approved"] else "❌"
            print(f"   {status} {entry['action']}")
        
        print("\n" + "="*60)
        print("✅ Safe computer control demonstrated!")
        print("="*60)


# Demo
if __name__ == "__main__":
    print("Choose permission level:")
    print("1. READ_ONLY - Can only observe")
    print("2. SUPERVISED - Requires confirmation (recommended)")
    print("3. SANDBOXED - Limited to sandbox")
    print("4. TRUSTED - Full access (dangerous!)")
    
    choice = input("\nChoice (1-4): ")
    
    levels = {
        "1": PermissionLevel.READ_ONLY,
        "2": PermissionLevel.SUPERVISED,
        "3": PermissionLevel.SANDBOXED,
        "4": PermissionLevel.TRUSTED
    }
    
    level = levels.get(choice, PermissionLevel.SUPERVISED)
    
    agent = SafeComputerAgent(permission_level=level)
    agent.demonstrate_safe_workflow()