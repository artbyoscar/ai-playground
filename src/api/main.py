# src/api/main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import os
import sys
from datetime import datetime
import redis
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing AI components
try:
    from core.working_ai_playground import AIPlayground
    from agents.research_specialist import ResearchSpecialist
    from agents.analyst import Analyst
    from agents.writer import Writer
    from agents.coder import Coder
except ImportError as e:
    print(f"Warning: Could not import AI components: {e}")

# Redis connection for conversation memory
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=6379,
        decode_responses=True
    )
    redis_client.ping()
    print("✅ Connected to Redis")
except:
    redis_client = None
    print("⚠️ Redis not available - conversation memory disabled")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 EdgeMind API starting up...")
    # Initialize your AI models here if needed
    yield
    # Shutdown
    print("👋 EdgeMind API shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="EdgeMind API",
    description="Open-source privacy-first AI platform API",
    version="0.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI components
try:
    ai_playground = AIPlayground()
    agents = {
        "research": ResearchSpecialist(),
        "analyst": Analyst(),
        "writer": Writer(),
        "coder": Coder()
    }
    print("✅ AI components initialized")
except:
    ai_playground = None
    agents = {}
    print("⚠️ AI components not available")

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    agent: Optional[str] = "research"
    conversation_id: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

class ChatResponse(BaseModel):
    response: str
    agent: str
    conversation_id: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str

# Conversation Memory Manager
class ConversationMemory:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_memory = {}  # Fallback if Redis not available
    
    async def save_message(self, conversation_id: str, message: ChatMessage):
        """Save a message to conversation history"""
        key = f"conversation:{conversation_id}"
        message_data = message.dict()
        message_data['timestamp'] = datetime.now().isoformat()
        
        if self.redis:
            # Save to Redis
            self.redis.rpush(key, json.dumps(message_data))
            self.redis.expire(key, 86400 * 7)  # Expire after 7 days
        else:
            # Fallback to local memory
            if conversation_id not in self.local_memory:
                self.local_memory[conversation_id] = []
            self.local_memory[conversation_id].append(message_data)
    
    async def get_conversation(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        key = f"conversation:{conversation_id}"
        
        if self.redis:
            messages = self.redis.lrange(key, 0, -1)
            return [ChatMessage(**json.loads(msg)) for msg in messages]
        else:
            messages = self.local_memory.get(conversation_id, [])
            return [ChatMessage(**msg) for msg in messages]
    
    async def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if self.redis:
            self.redis.delete(f"conversation:{conversation_id}")
        else:
            self.local_memory.pop(conversation_id, None)
    
    async def list_conversations(self) -> List[str]:
        """List all conversation IDs"""
        if self.redis:
            keys = self.redis.keys("conversation:*")
            return [key.split(":")[1] for key in keys]
        else:
            return list(self.local_memory.keys())

# Initialize conversation memory
memory = ConversationMemory(redis_client)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EdgeMind API",
        "version": "0.2.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "conversations": "/api/conversations",
            "agents": "/api/agents",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "redis": "connected" if redis_client else "disconnected",
        "ai": "ready" if ai_playground else "not initialized",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/agents")
async def list_agents():
    """List available AI agents"""
    return {
        "agents": [
            {
                "id": "research",
                "name": "Research Specialist",
                "description": "Multi-source web research and fact-checking"
            },
            {
                "id": "analyst",
                "name": "Data Analyst",
                "description": "Data processing and insights generation"
            },
            {
                "id": "writer",
                "name": "Content Writer",
                "description": "Creative and technical writing"
            },
            {
                "id": "coder",
                "name": "Code Assistant",
                "description": "Programming help and code generation"
            }
        ]
    }

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint"""
    if not ai_playground:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"
    
    # Save user message to memory
    user_message = ChatMessage(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    await memory.save_message(conversation_id, user_message)
    
    # Get conversation history for context
    history = await memory.get_conversation(conversation_id)
    
    # Select agent
    agent = agents.get(request.agent, agents.get("research"))
    
    # Generate response (simplified - you'd integrate your actual AI here)
    try:
        # Build context from history
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-5:]])
        
        # Generate response using your AI
        # response = await agent.process(request.message, context=context)
        response = f"[{request.agent}] Response to: {request.message}"  # Placeholder
        
        # Save assistant message to memory
        assistant_message = ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
            metadata={"agent": request.agent}
        )
        await memory.save_message(conversation_id, assistant_message)
        
        return ChatResponse(
            response=response,
            agent=request.agent,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def list_conversations():
    """List all conversations"""
    conversation_ids = await memory.list_conversations()
    return {"conversations": conversation_ids}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get specific conversation history"""
    messages = await memory.get_conversation(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=messages,
        created_at=messages[0].timestamp if messages else "",
        updated_at=messages[-1].timestamp if messages else ""
    )

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    await memory.clear_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}

# WebSocket for real-time streaming
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    conversation_id = f"ws_{datetime.now().timestamp()}"
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process message
            user_message = ChatMessage(
                role="user",
                content=data.get("message", ""),
                timestamp=datetime.now().isoformat()
            )
            await memory.save_message(conversation_id, user_message)
            
            # Stream response back
            # This is where you'd integrate streaming from your AI
            response = f"Echo: {data.get('message', '')}"
            
            await websocket.send_json({
                "type": "response",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save response
            assistant_message = ChatMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now().isoformat()
            )
            await memory.save_message(conversation_id, assistant_message)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {conversation_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)