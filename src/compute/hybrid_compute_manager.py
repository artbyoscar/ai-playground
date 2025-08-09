# src/compute/hybrid_compute_manager.py
"""
Hybrid Cloud-Local Compute Manager
Seamlessly switch between local and cloud resources while maintaining local-first principles
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import requests
from datetime import datetime

class HybridComputeManager:
    """
    Manages hybrid cloud-local compute resources for AI training and research.
    Prioritizes local computation, falls back to cloud when needed.
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.local_resources = self._check_local_resources()
        self.cloud_providers = self._initialize_cloud_providers()
        
    def _load_config(self) -> dict:
        """Load compute configuration"""
        return {
            # Local-first configuration
            "prefer_local": True,
            "local_gpu_threshold": 2.0,  # GB VRAM threshold for local processing
            
            # Cloud providers (in order of preference/cost)
            "cloud_providers": {
                "runpod": {
                    "enabled": True,
                    "api_key": os.getenv("RUNPOD_API_KEY", ""),
                    "max_cost_per_hour": 0.50,
                    "preferred_gpus": ["3060", "3070", "3080"],
                    "endpoint": "https://api.runpod.io/v2"
                },
                "vast_ai": {
                    "enabled": True,
                    "api_key": os.getenv("VAST_AI_API_KEY", ""),
                    "max_cost_per_hour": 0.40,
                    "preferred_gpus": ["3060", "3070"],
                    "endpoint": "https://vast.ai/api/v0"
                },
                "google_colab": {
                    "enabled": True,
                    "free_tier": True,
                    "pro_subscription": False,
                    "max_runtime_hours": 12
                },
                "kaggle": {
                    "enabled": True,
                    "api_key": os.getenv("KAGGLE_KEY", ""),
                    "free_gpu_hours": 30,  # Per week
                    "endpoint": "https://www.kaggle.com/api/v1"
                },
                "modal": {
                    "enabled": True,
                    "api_key": os.getenv("MODAL_TOKEN", ""),
                    "free_credits": 30,  # Monthly
                    "endpoint": "https://modal.com"
                }
            },
            
            # Task routing configuration
            "task_routing": {
                "small_models": "local",  # < 1GB models
                "medium_models": "hybrid",  # 1-5GB models
                "large_models": "cloud",  # > 5GB models
                "inference": "local",  # Always local with EdgeFormer
                "training": "hybrid",  # Local for small, cloud for large
                "research": "local"  # Web research stays local
            }
        }
    
    def _check_local_resources(self) -> dict:
        """Check available local compute resources"""
        import psutil
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            else:
                gpu_name = "No GPU"
                gpu_memory = 0
        except:
            cuda_available = False
            gpu_name = "No GPU"
            gpu_memory = 0
        
        return {
            "cpu_cores": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / 1e9,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
            "disk_space_gb": psutil.disk_usage('/').free / 1e9
        }
    
    def _initialize_cloud_providers(self) -> dict:
        """Initialize cloud provider connections"""
        providers = {}
        
        # RunPod.io (cheapest GPU rental)
        if self.config["cloud_providers"]["runpod"]["enabled"]:
            providers["runpod"] = RunPodProvider(
                api_key=self.config["cloud_providers"]["runpod"]["api_key"]
            )
        
        # Vast.ai (competitive GPU rental)
        if self.config["cloud_providers"]["vast_ai"]["enabled"]:
            providers["vast_ai"] = VastAIProvider(
                api_key=self.config["cloud_providers"]["vast_ai"]["api_key"]
            )
        
        # Google Colab (free tier)
        if self.config["cloud_providers"]["google_colab"]["enabled"]:
            providers["google_colab"] = ColabProvider()
        
        # Kaggle (free GPU hours)
        if self.config["cloud_providers"]["kaggle"]["enabled"]:
            providers["kaggle"] = KaggleProvider(
                api_key=self.config["cloud_providers"]["kaggle"]["api_key"]
            )
        
        # Modal (serverless)
        if self.config["cloud_providers"]["modal"]["enabled"]:
            providers["modal"] = ModalProvider(
                api_key=self.config["cloud_providers"]["modal"]["api_key"]
            )
        
        return providers
    
    def route_task(self, task_type: str, model_size_gb: float) -> str:
        """Determine where to run a task based on resources"""
        
        # Check if we can run locally
        if self.can_run_locally(model_size_gb):
            return "local"
        
        # Check task routing rules
        if task_type in self.config["task_routing"]:
            route = self.config["task_routing"][task_type]
            if route == "local" and not self.can_run_locally(model_size_gb):
                print(f"⚠️ Task prefers local but insufficient resources. Using cloud.")
                route = "cloud"
            return route
        
        # Default routing based on size
        if model_size_gb < 1:
            return "local"
        elif model_size_gb < 5:
            return "hybrid"
        else:
            return "cloud"
    
    def can_run_locally(self, model_size_gb: float) -> bool:
        """Check if a model can run locally"""
        required_ram = model_size_gb * 2  # Rule of thumb: 2x model size
        
        if self.local_resources["cuda_available"]:
            return self.local_resources["gpu_memory_gb"] >= model_size_gb
        else:
            return self.local_resources["ram_gb"] >= required_ram
    
    def get_cheapest_cloud_option(self, requirements: dict) -> dict:
        """Find the cheapest cloud option for given requirements"""
        options = []
        
        # Check each provider
        for provider_name, provider in self.cloud_providers.items():
            if provider.can_meet_requirements(requirements):
                cost = provider.estimate_cost(requirements)
                options.append({
                    "provider": provider_name,
                    "cost_per_hour": cost,
                    "setup_time": provider.setup_time_minutes,
                    "free_tier": provider.has_free_tier
                })
        
        # Sort by cost (free tier first, then cheapest)
        options.sort(key=lambda x: (not x["free_tier"], x["cost_per_hour"]))
        
        return options[0] if options else None
    
    def execute_training(self, 
                        model_config: dict,
                        dataset_path: str,
                        output_dir: str,
                        prefer_free: bool = True) -> dict:
        """Execute training with optimal compute allocation"""
        
        model_size_gb = model_config.get("size_gb", 1.0)
        task_type = "training"
        
        # Determine where to run
        route = self.route_task(task_type, model_size_gb)
        
        if route == "local":
            return self._train_locally(model_config, dataset_path, output_dir)
        elif route == "cloud":
            return self._train_on_cloud(model_config, dataset_path, output_dir, prefer_free)
        else:  # hybrid
            return self._train_hybrid(model_config, dataset_path, output_dir)
    
    def _train_locally(self, model_config: dict, dataset_path: str, output_dir: str) -> dict:
        """Train model locally"""
        print("🏠 Training locally...")
        
        # Import your EdgeFormer for compression
        from src.edgeformer_integration import EdgeFormerOptimizer
        optimizer = EdgeFormerOptimizer()
        
        # Compress model if needed to fit in local memory
        if not self.can_run_locally(model_config["size_gb"]):
            print("📦 Compressing model with EdgeFormer to fit local resources...")
            model_config = optimizer.compress_for_local(model_config)
        
        # Local training code here
        result = {
            "status": "success",
            "location": "local",
            "duration_minutes": 0,
            "cost": 0,
            "model_path": output_dir
        }
        
        return result
    
    def _train_on_cloud(self, 
                       model_config: dict, 
                       dataset_path: str, 
                       output_dir: str,
                       prefer_free: bool = True) -> dict:
        """Train model on cloud"""
        
        requirements = {
            "gpu_memory_gb": model_config["size_gb"] * 2,
            "duration_hours": model_config.get("estimated_hours", 1),
            "prefer_free": prefer_free
        }
        
        # Get cheapest option
        best_option = self.get_cheapest_cloud_option(requirements)
        
        if not best_option:
            raise ValueError("No suitable cloud provider found")
        
        print(f"☁️ Training on {best_option['provider']} "
              f"(${best_option['cost_per_hour']:.2f}/hour)")
        
        provider = self.cloud_providers[best_option["provider"]]
        
        # Upload data and start training
        job_id = provider.start_training(model_config, dataset_path)
        
        # Monitor progress
        result = provider.monitor_job(job_id)
        
        # Download results
        provider.download_results(job_id, output_dir)
        
        return result
    
    def _train_hybrid(self, model_config: dict, dataset_path: str, output_dir: str) -> dict:
        """Hybrid training - preprocess locally, train on cloud, fine-tune locally"""
        print("🔄 Hybrid training mode...")
        
        # Step 1: Preprocess and prepare data locally
        print("1️⃣ Preprocessing data locally...")
        preprocessed_data = self._preprocess_locally(dataset_path)
        
        # Step 2: Initial training on cloud (if needed)
        if model_config["size_gb"] > 2:
            print("2️⃣ Initial training on cloud...")
            cloud_result = self._train_on_cloud(
                model_config, 
                preprocessed_data, 
                output_dir,
                prefer_free=True
            )
            checkpoint_path = cloud_result["model_path"]
        else:
            checkpoint_path = None
        
        # Step 3: Fine-tune or compress locally with EdgeFormer
        print("3️⃣ Fine-tuning and compressing locally with EdgeFormer...")
        from src.edgeformer_integration import EdgeFormerOptimizer
        optimizer = EdgeFormerOptimizer()
        
        final_model = optimizer.optimize_and_compress(
            checkpoint_path or model_config,
            target_device="edge"
        )
        
        return {
            "status": "success",
            "location": "hybrid",
            "model_path": final_model,
            "optimized": True
        }
    
    def _preprocess_locally(self, dataset_path: str) -> str:
        """Preprocess data locally"""
        # Implement data preprocessing
        return dataset_path


class RunPodProvider:
    """RunPod.io GPU rental provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.runpod.io/v2"
        self.setup_time_minutes = 5
        self.has_free_tier = False
    
    def can_meet_requirements(self, requirements: dict) -> bool:
        """Check if RunPod can meet requirements"""
        # RunPod has GPUs from 8GB to 80GB VRAM
        return requirements["gpu_memory_gb"] <= 80
    
    def estimate_cost(self, requirements: dict) -> float:
        """Estimate cost per hour"""
        gpu_memory = requirements["gpu_memory_gb"]
        
        # RunPod pricing (approximate)
        if gpu_memory <= 12:  # RTX 3060
            return 0.24
        elif gpu_memory <= 24:  # RTX 3090
            return 0.44
        elif gpu_memory <= 48:  # RTX A6000
            return 0.79
        else:  # A100 80GB
            return 1.89
    
    def start_training(self, model_config: dict, dataset_path: str) -> str:
        """Start training job on RunPod"""
        # Implementation for RunPod API
        job_id = f"runpod_{datetime.now().timestamp()}"
        return job_id
    
    def monitor_job(self, job_id: str) -> dict:
        """Monitor training job"""
        return {"status": "completed", "job_id": job_id}
    
    def download_results(self, job_id: str, output_dir: str):
        """Download training results"""
        pass


class VastAIProvider:
    """Vast.ai GPU rental provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://vast.ai/api/v0"
        self.setup_time_minutes = 10
        self.has_free_tier = False
    
    def can_meet_requirements(self, requirements: dict) -> bool:
        return requirements["gpu_memory_gb"] <= 48
    
    def estimate_cost(self, requirements: dict) -> float:
        """Vast.ai is usually 20-30% cheaper than RunPod"""
        gpu_memory = requirements["gpu_memory_gb"]
        
        if gpu_memory <= 12:
            return 0.18
        elif gpu_memory <= 24:
            return 0.35
        else:
            return 0.65


class ColabProvider:
    """Google Colab provider (free tier available)"""
    
    def __init__(self):
        self.setup_time_minutes = 2
        self.has_free_tier = True
    
    def can_meet_requirements(self, requirements: dict) -> bool:
        # Colab free tier has limited GPU (usually T4 with 15GB)
        return requirements["gpu_memory_gb"] <= 15 and requirements.get("prefer_free", False)
    
    def estimate_cost(self, requirements: dict) -> float:
        return 0.0  # Free tier


class KaggleProvider:
    """Kaggle notebooks provider (30 free GPU hours/week)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.setup_time_minutes = 3
        self.has_free_tier = True
    
    def can_meet_requirements(self, requirements: dict) -> bool:
        # Kaggle provides P100 GPUs with 16GB VRAM
        return requirements["gpu_memory_gb"] <= 16 and requirements.get("prefer_free", False)
    
    def estimate_cost(self, requirements: dict) -> float:
        return 0.0  # Free tier (30 hours/week)


class ModalProvider:
    """Modal serverless compute provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.setup_time_minutes = 1
        self.has_free_tier = True  # $30 free credits/month
    
    def can_meet_requirements(self, requirements: dict) -> bool:
        return requirements["gpu_memory_gb"] <= 80
    
    def estimate_cost(self, requirements: dict) -> float:
        # Modal pricing
        gpu_memory = requirements["gpu_memory_gb"]
        if gpu_memory <= 16:
            return 0.59  # T4
        elif gpu_memory <= 40:
            return 1.10  # A10G
        else:
            return 2.89  # A100


def setup_cloud_providers():
    """Quick setup for cloud providers"""
    
    print("🌩️ CLOUD COMPUTE SETUP")
    print("=" * 50)
    print("Choose your cloud providers (multiple selections allowed):")
    print()
    print("FREE TIERS:")
    print("1. Google Colab (Free GPU, no setup)")
    print("2. Kaggle (30 GPU hours/week free)")
    print("3. Modal ($30 free credits/month)")
    print()
    print("PAID OPTIONS (cheapest):")
    print("4. RunPod.io ($0.24/hour for RTX 3060)")
    print("5. Vast.ai ($0.18/hour for RTX 3060)")
    print()
    
    # Save configuration
    config = {
        "google_colab": input("Use Google Colab? (y/n): ").lower() == 'y',
        "kaggle": input("Use Kaggle? (y/n): ").lower() == 'y',
        "modal": input("Use Modal? (y/n): ").lower() == 'y',
        "runpod": input("Use RunPod? (y/n): ").lower() == 'y',
        "vast_ai": input("Use Vast.ai? (y/n): ").lower() == 'y'
    }
    
    # Get API keys for selected providers
    env_content = []
    
    if config["kaggle"]:
        print("\n📝 Kaggle Setup:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create New API Token")
        print("3. Download kaggle.json")
        kaggle_key = input("Enter Kaggle username: ")
        if kaggle_key:
            env_content.append(f"KAGGLE_USERNAME={kaggle_key}")
            kaggle_secret = input("Enter Kaggle key: ")
            env_content.append(f"KAGGLE_KEY={kaggle_secret}")
    
    if config["modal"]:
        print("\n📝 Modal Setup:")
        print("1. Go to https://modal.com")
        print("2. Sign up for free account")
        print("3. Get API token from dashboard")
        modal_token = input("Enter Modal token (or press Enter to skip): ")
        if modal_token:
            env_content.append(f"MODAL_TOKEN={modal_token}")
    
    if config["runpod"]:
        print("\n📝 RunPod Setup:")
        print("1. Go to https://runpod.io")
        print("2. Create account and add credits")
        print("3. Get API key from settings")
        runpod_key = input("Enter RunPod API key (or press Enter to skip): ")
        if runpod_key:
            env_content.append(f"RUNPOD_API_KEY={runpod_key}")
    
    if config["vast_ai"]:
        print("\n📝 Vast.ai Setup:")
        print("1. Go to https://vast.ai")
        print("2. Create account and add credits")
        print("3. Get API key from account page")
        vast_key = input("Enter Vast.ai API key (or press Enter to skip): ")
        if vast_key:
            env_content.append(f"VAST_AI_API_KEY={vast_key}")
    
    # Save to .env file
    if env_content:
        with open(".env", "a") as f:
            f.write("\n# Cloud Compute Providers\n")
            f.write("\n".join(env_content))
            f.write("\n")
        print("\n✅ Cloud providers configured and saved to .env")
    
    return config


if __name__ == "__main__":
    # Quick test
    print("🚀 Hybrid Compute Manager Test")
    print("=" * 50)
    
    manager = HybridComputeManager()
    
    print("\n📊 Local Resources:")
    for key, value in manager.local_resources.items():
        print(f"  {key}: {value}")
    
    print("\n☁️ Cloud Providers Available:")
    for provider in manager.cloud_providers:
        print(f"  ✅ {provider}")
    
    # Test routing
    print("\n🎯 Task Routing Tests:")
    test_cases = [
        ("training", 0.5),
        ("training", 3.0),
        ("training", 10.0),
        ("inference", 1.0),
        ("research", 0.1)
    ]
    
    for task_type, size_gb in test_cases:
        route = manager.route_task(task_type, size_gb)
        print(f"  {task_type} ({size_gb}GB): → {route}")
    
    # Setup cloud providers if needed
    setup_option = input("\n🔧 Setup cloud providers now? (y/n): ")
    if setup_option.lower() == 'y':
        setup_cloud_providers()