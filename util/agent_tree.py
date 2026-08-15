from openai import OpenAI
import base64
import os
import logging as logger
import time

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_mime_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/png') 


def make_content(text,role="user",images_url=None):
    content=[]
    content.append({
        "text": text,
        "type": "text"
    })
    if images_url is not None:
        for img_url in images_url:
            mime_type = get_image_mime_type(img_url)
            content.append({
                "image_url": {
                    "url": f"data:{mime_type};base64,{encode_image(img_url)}"
                },
                "type": "image_url"
            })
    return {'role': role, 'content': content}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Agent:
    def __init__(self,api,url,system_message=None,use_history=True,max_retries=3, retry_delay=5, timeout=120):
        self.model = os.getenv("RECONTRASTER_AGENT_MODEL", "gpt-4o")
        timeout = _env_float("RECONTRASTER_AGENT_TIMEOUT", timeout)
        max_retries = _env_int("RECONTRASTER_AGENT_MAX_RETRIES", max_retries)
        retry_delay = _env_float("RECONTRASTER_AGENT_RETRY_DELAY", retry_delay)
        self.client= OpenAI(
            api_key=api,
            base_url=url,
            timeout=timeout,
            max_retries=0,
        )
        self.context=[]
        self.use_history = use_history
        self.use_history = use_history
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if system_message:
            self.context.append({
                "role": "system",
                "content": system_message
            })

    def _make_api_call(self, messages):
        """执行API调用的核心方法，包含重试逻辑"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"尝试API调用 (第 {attempt + 1}/{self.max_retries} 次)")
                
                response = self.client.chat.completions.create(
                    messages=messages, 
                    model=self.model, 
                    stream=False
                )
                
                logger.info("API调用成功")
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                
                logger.warning(f"第 {attempt + 1} 次尝试失败: {error_type} - {error_msg}")
                
                # 判断错误类型
                if "timeout" in error_msg.lower() or "524" in error_msg:
                    logger.info("检测到超时错误，将进行重试")
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    logger.info("检测到速率限制，延长等待时间")
                    time.sleep(self.retry_delay * 2)  # 速率限制时等待更长时间
                elif "500" in error_msg or "502" in error_msg or "503" in error_msg:
                    logger.info("检测到服务器错误，将进行重试")
                else:
                    logger.error(f"未知错误类型: {error_type}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("所有重试尝试都失败了")
        
        # 所有重试都失败，抛出最后一个异常
        logger.error(f"API调用最终失败: {type(last_exception).__name__} - {last_exception}")
        raise last_exception

    def chat_with_history(self, text, role="user", images_url=None):
        """带历史记录的聊天"""
        try:
            content = make_content(text, role, images_url)
            self.context.append(content)
            
            result = self._make_api_call(self.context)
            
            self.context.append({'role': 'assistant', 'content': result})
            return result
            
        except Exception as e:
            # 如果出错，移除刚添加的用户消息
            if self.context and self.context[-1].get('role') == role:
                self.context.pop()
            logger.error(f"chat_with_history 失败: {e}")
            raise

    def chat_with_out_history(self, text, role="user", images_url=None):
        """不带历史记录的聊天"""
        try:
            content = make_content(text, role, images_url)
            context = self.context.copy()
            context.append(content)
            
            return self._make_api_call(context)
            
        except Exception as e:
            logger.error(f"chat_with_out_history 失败: {e}")
            raise

    def chat(self, text, role="user", images_url=None):
        """主聊天方法，根据配置选择是否使用历史记录"""
        try:
            if self.use_history:
                return self.chat_with_history(text, role, images_url)
            else:
                return self.chat_with_out_history(text, role, images_url)
        except Exception as e:
            logger.error(f"聊天请求失败: {e}")
            raise
    
            
def single_chat(api, url, text, role="user", images_url=None, system_message=None):
    agent = Agent(api, url, system_message)
    return agent.chat(text, role, images_url)
