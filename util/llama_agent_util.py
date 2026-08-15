import os
import torch
import logging as logger
import time
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Dict, Optional, List, Union


class LlamaVisionAgent:
    """使用 Llama Vision 模型的 Agent 类（无历史记录版本）"""
    
    def __init__(self, model, processor, system_message=None, max_retries=3, retry_delay=5):
        """
        初始化 Llama Vision Agent
        
        参数:
            model: 已加载的模型实例
            processor: 已加载的处理器实例
            system_message: 系统提示消息（可选）
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间(秒)
        """
        self.model = model
        self.processor = processor
        self.system_message = system_message
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _load_images(self, image_paths):
        """加载图片文件"""
        if image_paths is None:
            return None
        
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
                logger.info(f"成功加载图片: {path}")
            except Exception as e:
                logger.error(f"加载图片失败 {path}: {e}")
                raise
        
        return images if images else None
    
    def _make_messages(self, text, images=None):
        """构建消息列表"""
        messages = []
        
        # 添加系统消息（如果有）
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        # 构建用户消息内容
        content = []
        
        # 添加图片
        if images:
            for _ in images:
                content.append({"type": "image"})
        
        # 添加文本
        content.append({"type": "text", "text": text})
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _generate_response(self, messages, images=None):
        """生成模型响应的核心方法，包含重试逻辑"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"尝试生成响应 (第 {attempt + 1}/{self.max_retries} 次)")
                
                # 应用聊天模板
                input_text = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True
                )
                
                # 准备输入
                if images:
                    inputs = self.processor(
                        images,
                        input_text,
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).to(self.model.device)
                else:
                    inputs = self.processor(
                        input_text,
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).to(self.model.device)
                
                # 生成响应
                output = self.model.generate(**inputs,max_new_tokens=4096,do_sample=False,num_beams=1, )
                
                # 解码输出
                full_response = self.processor.decode(output[0], skip_special_tokens=True)
                
                # 提取助手回复部分
                if "assistant" in full_response.lower():
                    response = full_response[full_response.lower().rfind("assistant") + len("assistant"):].strip()
                else:
                    response = full_response.strip()
                
                logger.info("响应生成成功")
                return response
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                
                logger.warning(f"第 {attempt + 1} 次尝试失败: {error_type} - {error_msg}")
                
                # 判断错误类型
                if "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
                    logger.error("检测到显存不足错误")
                    if attempt < self.max_retries - 1:
                        logger.info("尝试清理显存...")
                        torch.cuda.empty_cache()
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("所有重试尝试都失败了")
        
        # 所有重试都失败，抛出最后一个异常
        logger.error(f"生成响应最终失败: {type(last_exception).__name__} - {last_exception}")
        raise last_exception
    
    def chat(self, text, image_paths=None):
        """
        主聊天方法（无历史记录）
        
        参数:
            text: 用户输入文本
            image_paths: 图片路径列表或单个路径
            
        返回:
            模型的响应文本
        """
        try:
            # 加载图片
            images = self._load_images(image_paths)
            
            # 构建消息
            messages = self._make_messages(text, images)
            
            # 生成响应
            return self._generate_response(messages, images)
            
        except Exception as e:
            logger.error(f"聊天请求失败: {e}")
            raise


class MultiAgentManager:
    """多 Agent 管理器 - 管理多个共享同一模型的 Agent"""
    
    def __init__(self, model_path: str):
        """
        初始化多 Agent 管理器
        
        参数:
            model_path: Llama Vision 模型路径
        """
        logger.info(f"正在加载共享模型: {model_path}")
        
        # 加载模型和处理器（所有 agent 共享）
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        logger.info("共享模型加载完成")
        
        # 存储所有 agent
        self.agents: Dict[str, LlamaVisionAgent] = {}
    
    def create_agent(
        self, 
        agent_name: str, 
        system_message: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> LlamaVisionAgent:
        """
        创建一个新的 agent
        
        参数:
            agent_name: Agent 的名称（唯一标识）
            system_message: 系统提示消息
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间
            
        返回:
            创建的 Agent 实例
        """
        if agent_name in self.agents:
            logger.warning(f"Agent '{agent_name}' 已存在，将被覆盖")
        
        agent = LlamaVisionAgent(
            model=self.model,
            processor=self.processor,
            system_message=system_message,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        self.agents[agent_name] = agent
        logger.info(f"创建 Agent: {agent_name}")
        
        return agent
    
    def get_agent(self, agent_name: str) -> Optional[LlamaVisionAgent]:
        """
        获取指定名称的 agent
        
        参数:
            agent_name: Agent 的名称
            
        返回:
            Agent 实例，如果不存在则返回 None
        """
        return self.agents.get(agent_name)
    
    def remove_agent(self, agent_name: str) -> bool:
        """
        删除指定的 agent
        
        参数:
            agent_name: Agent 的名称
            
        返回:
            是否成功删除
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"删除 Agent: {agent_name}")
            return True
        else:
            logger.warning(f"Agent '{agent_name}' 不存在")
            return False
    
    def list_agents(self) -> List[str]:
        """
        列出所有 agent 的名称
        
        返回:
            Agent 名称列表
        """
        return list(self.agents.keys())
    
    def chat(self, agent_name: str, text: str, image_paths: Optional[Union[str, List[str]]] = None) -> str:
        """
        使用指定的 agent 进行对话
        
        参数:
            agent_name: Agent 的名称
            text: 用户输入文本
            image_paths: 图片路径列表或单个路径
            
        返回:
            模型的响应文本
        """
        agent = self.get_agent(agent_name)
        
        if agent is None:
            raise ValueError(f"Agent '{agent_name}' 不存在")
        
        return agent.chat(text, image_paths)
    
    def clear_cache(self):
        """清理 GPU 缓存"""
        torch.cuda.empty_cache()
        logger.info("GPU 缓存已清理")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 模型路径
    model_path = os.getenv("LLAMA_MODEL_PATH")
    if not model_path:
        raise RuntimeError("请设置 LLAMA_MODEL_PATH")
    
    # 创建多 Agent 管理器
    print("=" * 50)
    print("创建多 Agent 管理器")
    print("=" * 50)
    
    manager = MultiAgentManager(model_path)
    
    # 创建多个不同用途的 agent
    print("\n创建不同的 Agents...")
    
    # Agent 1: 图像描述专家
    manager.create_agent(
        agent_name="image_describer",
        system_message="You are an expert in describing images in great detail."
    )
    
    # Agent 2: 对象识别专家
    manager.create_agent(
        agent_name="object_detector",
        system_message="You are an expert in identifying and listing objects in images."
    )
    
    # Agent 3: 图像比较专家
    manager.create_agent(
        agent_name="image_comparator",
        system_message="You are an expert in comparing and analyzing differences between images."
    )
    
    # Agent 4: 通用助手（无特殊系统消息）
    manager.create_agent(
        agent_name="general_assistant"
    )
    
    # 列出所有 agent
    print(f"\n当前的 Agents: {manager.list_agents()}")
    
    # 使用不同的 agent 处理任务
    print("\n" + "=" * 50)
    print("使用不同的 Agents 处理任务")
    print("=" * 50)
    
    # 使用 image_describer
    print("\n[使用 image_describer]")
    response1 = manager.chat(
        agent_name="image_describer",
        text="Describe this image in detail.",
        image_paths=os.getenv("LLAMA_IMAGE_1")
    )
    print(f"响应: {response1}\n")
    
    # 使用 object_detector
    print("\n[使用 object_detector]")
    response2 = manager.chat(
        agent_name="object_detector",
        text="List all objects you can identify in this image.",
        image_paths=os.getenv("LLAMA_IMAGE_1")
    )
    print(f"响应: {response2}\n")
    
    # 使用 image_comparator 比较两张图片
    print("\n[使用 image_comparator]")
    response3 = manager.chat(
        agent_name="image_comparator",
        text="What are the similarities and differences between these two images?",
        image_paths=[os.getenv("LLAMA_IMAGE_1"), os.getenv("LLAMA_IMAGE_2")]
    )
    print(f"响应: {response3}\n")
    
    # 使用 general_assistant
    print("\n[使用 general_assistant]")
    response4 = manager.chat(
        agent_name="general_assistant",
        text="What is in this image?",
        image_paths=os.getenv("LLAMA_IMAGE_2")
    )
    print(f"响应: {response4}\n")
    
    # 也可以直接获取 agent 实例使用
    print("\n" + "=" * 50)
    print("直接获取 Agent 实例使用")
    print("=" * 50)
    
    describer = manager.get_agent("image_describer")
    response5 = describer.chat(
        text="Focus on the colors in this image.",
        image_paths=os.getenv("LLAMA_IMAGE_1")
    )
    print(f"响应: {response5}\n")
    
    # 删除某个 agent
    print("\n删除 general_assistant...")
    manager.remove_agent("general_assistant")
    print(f"剩余的 Agents: {manager.list_agents()}")
