import requests
import json
import time
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import tempfile
import cv2


# 通用视频处理函数
def process_video_from_url(url, node_name="Video"):
    """从 URL下载并处理视频为帧序列和音频"""
    print(f"[{node_name}] 开始下载视频: {url}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file_path = temp_file.name
    
    print(f"[{node_name}] 视频下载完成，开始处理...")
    cap = cv2.VideoCapture(temp_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"[{node_name}] 视频信息: {width}x{height}, {fps}fps, {total_frames}帧, 时长{duration:.2f}秒")
    
    frames = []
    frame_count = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).float() / 255.0
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    # 提取音频
    audio_data = None
    try:
        import torchaudio
        import subprocess
        
        print(f"[{node_name}] 正在提取音频...")
        
        # 使用ffmpeg提取音频到临时WAV文件
        audio_temp_path = temp_file_path.replace('.mp4', '_audio.wav')
        
        # 尝试使用ffmpeg提取音频
        try:
            cmd = [
                'ffmpeg', '-i', temp_file_path,
                '-vn',  # 不包含视频
                '-acodec', 'pcm_s16le',  # 使用PCM编码
                '-ar', '44100',  # 采样率44100Hz
                '-ac', '2',  # 双声道
                '-y',  # 覆盖输出文件
                audio_temp_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            
            # 加载提取的音频
            waveform, sample_rate = torchaudio.load(audio_temp_path)
            audio_data = {'waveform': waveform.unsqueeze(0), 'sample_rate': sample_rate}
            print(f"[{node_name}] 音频提取成功: {waveform.shape}, 采样率: {sample_rate}Hz")
            
            # 删除临时音频文件
            try:
                os.unlink(audio_temp_path)
            except:
                pass
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[{node_name}] ffmpeg提取失败: {str(e)}，尝试直接加载...")
            # 如果ffmpeg失败，尝试直接用torchaudio加载
            try:
                waveform, sample_rate = torchaudio.load(temp_file_path)
                audio_data = {'waveform': waveform.unsqueeze(0), 'sample_rate': sample_rate}
                print(f"[{node_name}] 音频直接加载成功: {waveform.shape}, 采样率: {sample_rate}Hz")
            except:
                raise
                
    except Exception as e:
        print(f"[{node_name}] 音频提取失败: {str(e)}")
        # 如果提取音频失败，返回空音频
        audio_data = {'waveform': torch.zeros(1, 2, int(duration * 44100)), 'sample_rate': 44100}
        print(f"[{node_name}] 使用静音作为替代")
    
    os.unlink(temp_file_path)
    print(f"[{node_name}] 临时文件已删除")
    frames = torch.stack(frames)
    video_info = {"fps": fps, "frame_count": total_frames, "duration": duration, "width": width, "height": height}
    video_info_str = json.dumps(video_info, ensure_ascii=False, indent=2)
    print(f"[{node_name}] 处理完成! 共加载{frame_count}帧")
    return (frames, frame_count, video_info_str, audio_data)


class OAIQwenTextToImage:
    """OAI Qwen千问文生图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "ai_expansion": ("BOOLEAN", {
                    "default": False
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "OAI"
    
    def generate_image(self, api_key, prompt, aspect_ratio, batch_size, ai_expansion, seed):
        """生成图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 提交任务
        print(f"[OAI Qwen] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, aspect_ratio, batch_size, ai_expansion)
        print(f"[OAI Qwen] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Qwen] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Qwen] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                # 尝试获取更详细的错误信息
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Qwen] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                # 如果没有详细错误信息，给出可能的原因提示
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词包含敏感内容被审核拒绝 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                # 任务仍在处理中
                print(f"[OAI Qwen] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _submit_task(self, api_key, prompt, aspect_ratio, batch_size, ai_expansion):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "qwenshengtu",
            "parameter": {
                "prompt": prompt,
                "ai_expansion": ai_expansion,
                "aspect_ratio": aspect_ratio,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Qwen] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Qwen] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            # 检查返回数据结构
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            # API返回的是taskId（驼峰命名）或task_id
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 使用GET方法，taskId作为路径参数
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Qwen] 查询响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Qwen] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Qwen] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                # 将图像数据转换为PIL Image
                image = Image.open(io.BytesIO(response.content))
                
                # 转换为RGB模式（如果是RGBA或其他模式）
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 转换为numpy数组
                image_np = np.array(image).astype(np.float32) / 255.0
                
                # 转换为torch tensor
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Qwen] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Qwen] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIJimengTextToImage:
    """OAI 即梦绘画节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
                "use_pre_llm": ("BOOLEAN", {
                    "default": True
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "OAI"
    
    def generate_image(self, api_key, prompt, aspect_ratio, use_pre_llm, seed):
        """生成图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 提交任务
        print(f"[OAI Jimeng] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, aspect_ratio, use_pre_llm)
        print(f"[OAI Jimeng] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Jimeng] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Jimeng] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                # 尝试获取更详细的错误信息
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Jimeng] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                # 如果没有详细错误信息，给出可能的原因提示
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词包含敏感内容被审核拒绝 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                # 任务仍在处理中
                print(f"[OAI Jimeng] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _submit_task(self, api_key, prompt, aspect_ratio, use_pre_llm):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "jimeng",
            "parameter": {
                "prompt": prompt,
                "use_pre_llm": use_pre_llm,
                "aspect_ratio": aspect_ratio
            }
        }
        
        print(f"[OAI Jimeng] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Jimeng] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            # 检查返回数据结构
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            # API返回的是taskId（驼峰命名）或task_id
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 使用GET方法，taskId作为路径参数
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Jimeng] 查询响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Jimeng] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            # 将图像数据转换为PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # 转换为RGB模式（如果是RGBA或其他模式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为torch tensor，格式为 [batch, height, width, channels]
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Jimeng] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIWanwurongtu:
    """OAI AI万物溶图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, seed):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 将图片上传到OSS获取URL
        print(f"[OAI Wanwurongtu] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Wanwurongtu] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Wanwurongtu] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Wanwurongtu] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Wanwurongtu] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Wanwurongtu] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Wanwurongtu] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Wanwurongtu] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "wanwurongtu",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Wanwurongtu] 提交参数: appId={payload['appId']}, image_url={image_url}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wanwurongtu] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Wanwurongtu] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Wanwurongtu] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIZhengJianZhao:
    """OAI AI证件照节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "background_color": ("STRING", {
                    "default": "白色",
                    "multiline": False,
                    "placeholder": "请输入背景颜色（如：白色、蓝色、红色）"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, background_color, aspect_ratio):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not background_color or not background_color.strip():
            raise ValueError("背景颜色不能为空，请输入背景颜色")
        
        # 将图片上传到OSS获取URL
        print(f"[OAI ZhengJianZhao] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI ZhengJianZhao] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI ZhengJianZhao] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, background_color, aspect_ratio)
        print(f"[OAI ZhengJianZhao] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI ZhengJianZhao] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI ZhengJianZhao] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI ZhengJianZhao] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI ZhengJianZhao] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, background_color, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "AI证件照",
            "parameter": {
                "image": image_url,
                "background_color": background_color,
                "aspect_ratio": aspect_ratio
            }
        }
        
        print(f"[OAI ZhengJianZhao] 提交参数: appId={payload['appId']}, image_url={image_url}, background_color={background_color}, aspect_ratio={aspect_ratio}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI ZhengJianZhao] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI ZhengJianZhao] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI ZhengJianZhao] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
           raise Exception(f"处理图像失败: {str(e)}")


class OAIRenwuchangjingronghe:
    """OAI 人物场景融合节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, prompt, batch_size):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 上传两张图片到OSS
        print(f"[OAI Renwuchangjingronghe] 正在上传图片1到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Renwuchangjingronghe] 图片1上传成功: {image1_url}")
        
        print(f"[OAI Renwuchangjingronghe] 正在上传图片2到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Renwuchangjingronghe] 图片2上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Renwuchangjingronghe] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, prompt, batch_size)
        print(f"[OAI Renwuchangjingronghe] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Renwuchangjingronghe] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Renwuchangjingronghe] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Renwuchangjingronghe] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Renwuchangjingronghe] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, prompt, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "renwuchagjingronghe",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "prompt": prompt,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Renwuchangjingronghe] 提交参数: appId={payload['appId']}, image1={image1_url}, image2={image2_url}, prompt={prompt}, batch_size={batch_size}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Renwuchangjingronghe] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Renwuchangjingronghe] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Renwuchangjingronghe] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Renwuchangjingronghe] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Renwuchangjingronghe] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIZitaiqianyi:
    """OAI 人物姿态迁移节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, batch_size, aspect_ratio):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传两张图片到OSS
        print(f"[OAI Zitaiqianyi] 正在上传图片1到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Zitaiqianyi] 图片1上传成功: {image1_url}")
        
        print(f"[OAI Zitaiqianyi] 正在上传图片2到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Zitaiqianyi] 图片2上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Zitaiqianyi] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, batch_size, aspect_ratio)
        print(f"[OAI Zitaiqianyi] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Zitaiqianyi] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Zitaiqianyi] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Zitaiqianyi] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Zitaiqianyi] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, batch_size, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "zitaiqianyi",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "batch_size": str(batch_size),
                "aspect_ratio": aspect_ratio
            }
        }
        
        print(f"[OAI Zitaiqianyi] 提交参数: appId={payload['appId']}, image1={image1_url}, image2={image2_url}, batch_size={batch_size}, aspect_ratio={aspect_ratio}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Zitaiqianyi] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Zitaiqianyi] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Zitaiqianyi] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Zitaiqianyi] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Zitaiqianyi] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIQianzibaitai:
    """OAI 千姿百态节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "style_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入风格提示词"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, image3, style_prompt):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not style_prompt or not style_prompt.strip():
            raise ValueError("风格提示词不能为空，请输入风格提示词")
        
        # 上传三张图片到OSS
        print(f"[OAI Qianzibaitai] 正在上传图片1到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Qianzibaitai] 图片1上传成功: {image1_url}")
        
        print(f"[OAI Qianzibaitai] 正在上传图片2到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Qianzibaitai] 图片2上传成功: {image2_url}")
        
        print(f"[OAI Qianzibaitai] 正在上传图片3到OSS...")
        image3_url = self._upload_image_to_oss(image3)
        print(f"[OAI Qianzibaitai] 图片3上传成功: {image3_url}")
        
        # 提交任务
        print(f"[OAI Qianzibaitai] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, image3_url, style_prompt)
        print(f"[OAI Qianzibaitai] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Qianzibaitai] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Qianzibaitai] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Qianzibaitai] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Qianzibaitai] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, image3_url, style_prompt):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "qianzibaitai",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "image3": image3_url,
                "style_prompt": style_prompt
            }
        }
        
        print(f"[OAI Qianzibaitai] 提交参数: appId={payload['appId']}, image1={image1_url}, image2={image2_url}, image3={image3_url}, style_prompt={style_prompt}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Qianzibaitai] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Qianzibaitai] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Qianzibaitai] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIQwenEdit:
    """OAI Qwen编辑图像节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入编辑提示词"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, prompt, batch_size, aspect_ratio, image2=None, image3=None):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("编辑提示词不能为空，请输入编辑提示词")
        
        # 上传第一张图片到OSS（必须）
        print(f"[OAI QwenEdit] 正在上传图片1到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI QwenEdit] 图片1上传成功: {image1_url}")
        
        # 上传第二张图片到OSS（可选）
        image2_url = None
        if image2 is not None:
            print(f"[OAI QwenEdit] 正在上传图片2到OSS...")
            image2_url = self._upload_image_to_oss(image2)
            print(f"[OAI QwenEdit] 图片2上传成功: {image2_url}")
        
        # 上传第三张图片到OSS（可选）
        image3_url = None
        if image3 is not None:
            print(f"[OAI QwenEdit] 正在上传图片3到OSS...")
            image3_url = self._upload_image_to_oss(image3)
            print(f"[OAI QwenEdit] 图片3上传成功: {image3_url}")
        
        # 提交任务
        print(f"[OAI QwenEdit] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, image3_url, prompt, batch_size, aspect_ratio)
        print(f"[OAI QwenEdit] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI QwenEdit] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI QwenEdit] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI QwenEdit] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI QwenEdit] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, image3_url, prompt, batch_size, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建参数，只包含有值的图片
        parameter = {
            "image1": image1_url,
            "prompt": prompt,
            "batch_size": str(batch_size),
            "aspect_ratio": aspect_ratio
        }
        
        # 只在图片存在时添加到参数中
        if image2_url:
            parameter["image2"] = image2_url
        if image3_url:
            parameter["image3"] = image3_url
        
        payload = {
            "appId": "qwenedit",
            "parameter": parameter
        }
        
        print(f"[OAI QwenEdit] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI QwenEdit] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI QwenEdit] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI QwenEdit] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI QwenEdit] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI QwenEdit] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIFantuichutu:
    """OAI AI反推出图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入文本描述"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, text, batch_size):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not text or not text.strip():
            raise ValueError("文本描述不能为空，请输入文本描述")
        
        # 上传图片到OSS
        print(f"[OAI Fantuichutu] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Fantuichutu] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Fantuichutu] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, text, batch_size)
        print(f"[OAI Fantuichutu] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Fantuichutu] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Fantuichutu] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Fantuichutu] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Fantuichutu] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, text, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "qwenfantuichutu",
            "parameter": {
                "image": image_url,
                "text": text,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Fantuichutu] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Fantuichutu] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Fantuichutu] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Fantuichutu] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Fantuichutu] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Fantuichutu] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIXiangaochengtu:
    """OAI 线稿成图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, prompt):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 上传图片到OSS
        print(f"[OAI Xiangaochengtu] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Xiangaochengtu] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Xiangaochengtu] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, prompt)
        print(f"[OAI Xiangaochengtu] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Xiangaochengtu] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Xiangaochengtu] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Xiangaochengtu] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Xiangaochengtu] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, prompt):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "xiangaochengtu",
            "parameter": {
                "image": image_url,
                "prompt": prompt
            }
        }
        
        print(f"[OAI Xiangaochengtu] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Xiangaochengtu] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Xiangaochengtu] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Xiangaochengtu] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIChongdaguang:
    """OAI AI废片拯救（重打光）节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Chongdaguang] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Chongdaguang] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Chongdaguang] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Chongdaguang] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Chongdaguang] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Chongdaguang] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Chongdaguang] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Chongdaguang] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "chongdaguang",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Chongdaguang] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Chongdaguang] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Chongdaguang] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Chongdaguang] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIZhenrenshouban:
    """OAI 真人一键手办节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "strength": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, strength):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Zhenrenshouban] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Zhenrenshouban] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Zhenrenshouban] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, strength)
        print(f"[OAI Zhenrenshouban] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Zhenrenshouban] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Zhenrenshouban] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Zhenrenshouban] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Zhenrenshouban] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, strength):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "zhenrenyijianshouban",
            "parameter": {
                "image": image_url,
                "strength": str(strength)
            }
        }
        
        print(f"[OAI Zhenrenshouban] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Zhenrenshouban] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Zhenrenshouban] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Zhenrenshouban] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIQushuiyin:
    """OAI 智能消除（去水印）节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入要消除的内容描述（如：水印、文字等）"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, prompt):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入要消除的内容描述")
        
        # 上传图片到OSS
        print(f"[OAI Qushuiyin] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Qushuiyin] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Qushuiyin] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, prompt)
        print(f"[OAI Qushuiyin] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Qushuiyin] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Qushuiyin] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Qushuiyin] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Qushuiyin] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, prompt):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "qushuiyin",
            "parameter": {
                "image": image_url,
                "prompt": prompt
            }
        }
        
        print(f"[OAI Qushuiyin] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Qushuiyin] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Qushuiyin] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Qushuiyin] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAILaozhaopian:
    """OAI 老照片修复节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Laozhaopian] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Laozhaopian] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Laozhaopian] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Laozhaopian] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Laozhaopian] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Laozhaopian] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Laozhaopian] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Laozhaopian] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "laozhaopianxiufu",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Laozhaopian] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Laozhaopian] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Laozhaopian] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Laozhaopian] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIDianshang:
    """OAI 电商带货节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "aspect_ratio": (["9:16", "16:9", "1:1", "3:4", "4:3"], {
                    "default": "9:16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, prompt, batch_size, aspect_ratio):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 上传图片到OSS
        print(f"[OAI Dianshang] 正在上传第一张图片到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Dianshang] 第一张图片上传成功: {image1_url}")
        
        print(f"[OAI Dianshang] 正在上传第二张图片到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Dianshang] 第二张图片上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Dianshang] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, prompt, batch_size, aspect_ratio)
        print(f"[OAI Dianshang] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Dianshang] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Dianshang] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Dianshang] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Dianshang] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, prompt, batch_size, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "dianshangdaihuo",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "prompt": prompt,
                "batch_size": str(batch_size),
                "aspect_ratio": aspect_ratio
            }
        }
        
        print(f"[OAI Dianshang] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Dianshang] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Dianshang] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Dianshang] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Dianshang] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Dianshang] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIFlux:
    """OAI Flux图像编辑节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "mode": (["图像生成", "图像编辑", "文字生成"], {
                    "default": "图像生成"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "aspect_ratio": (["9:16", "16:9", "1:1", "3:4", "4:3"], {
                    "default": "9:16"
                }),
                "max": ("BOOLEAN", {
                    "default": False
                }),
                "text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入文字内容（可选）"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, mode, image, prompt, aspect_ratio, max, text, batch_size):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 上传图片到OSS
        print(f"[OAI Flux] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Flux] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Flux] 开始提交任务...")
        task_id = self._submit_task(api_key, mode, image_url, prompt, aspect_ratio, max, text, batch_size)
        print(f"[OAI Flux] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Flux] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Flux] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Flux] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Flux] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, mode, image_url, prompt, aspect_ratio, max, text, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "fluxtuxiangbianji",
            "parameter": {
                "mode": mode,
                "image": image_url,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "max": max,
                "text": text,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Flux] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Flux] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Flux] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Flux] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Flux] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Flux] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIKeling:
    """OAI 可灵绘图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "image": ("IMAGE",),
                "model": (["kling-v2-1", "kling-v1"], {
                    "default": "kling-v2-1"
                }),
                "mode": (["face", "style", "subject"], {
                    "default": "face"
                }),
                "strength1": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "strength2": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "aspect_ratio": (["9:16", "16:9", "1:1", "3:4", "4:3"], {
                    "default": "9:16"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, prompt, image, model, mode, strength1, strength2, aspect_ratio, batch_size):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 上传图片到OSS
        print(f"[OAI Keling] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Keling] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Keling] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, image_url, model, mode, strength1, strength2, aspect_ratio, batch_size)
        print(f"[OAI Keling] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Keling] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Keling] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Keling] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Keling] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, prompt, image_url, model, mode, strength1, strength2, aspect_ratio, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "kelingimage",
            "parameter": {
                "prompt": prompt,
                "image": image_url,
                "model": model,
                "mode": mode,
                "strength1": str(strength1),
                "strength2": str(strength2),
                "aspect_ratio": aspect_ratio,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Keling] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Keling] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Keling] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Keling] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Keling] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Keling] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIJipuli:
    """OAI 吉卜力风格图生图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Jipuli] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Jipuli] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Jipuli] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Jipuli] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Jipuli] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Jipuli] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Jipuli] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Jipuli] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "jipuli",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Jipuli] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Jipuli] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Jipuli] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Jipuli] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIChanpin:
    """OAI 产品重打光换背景v2节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Chanpin] 正在上传第一张图片到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Chanpin] 第一张图片上传成功: {image1_url}")
        
        print(f"[OAI Chanpin] 正在上传第二张图片到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Chanpin] 第二张图片上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Chanpin] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url)
        print(f"[OAI Chanpin] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Chanpin] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Chanpin] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Chanpin] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Chanpin] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "chanpinchongdaguang",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url
            }
        }
        
        print(f"[OAI Chanpin] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Chanpin] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Chanpin] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Chanpin] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIGaoqing:
    """OAI 高清放大节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Gaoqing] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Gaoqing] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Gaoqing] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Gaoqing] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Gaoqing] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Gaoqing] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Gaoqing] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Gaoqing] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "renxianggaoqingfangda",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Gaoqing] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Gaoqing] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Gaoqing] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Gaoqing] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIMaopei:
    """OAI 毛坯房装修v3节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, batch_size):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Maopei] 正在上传第一张图片到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Maopei] 第一张图片上传成功: {image1_url}")
        
        print(f"[OAI Maopei] 正在上传第二张图片到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Maopei] 第二张图片上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Maopei] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, batch_size)
        print(f"[OAI Maopei] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Maopei] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Maopei] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Maopei] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Maopei] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "maopeifzhuangxiu",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Maopei] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Maopei] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Maopei] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Maopei] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIWanwuqianyi:
    """OAI AI万物迁移v4节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, strength):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Wanwuqianyi] 正在上传第一张图片到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Wanwuqianyi] 第一张图片上传成功: {image1_url}")
        
        print(f"[OAI Wanwuqianyi] 正在上传第二张图片到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Wanwuqianyi] 第二张图片上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Wanwuqianyi] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, strength)
        print(f"[OAI Wanwuqianyi] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Wanwuqianyi] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Wanwuqianyi] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Wanwuqianyi] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Wanwuqianyi] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, strength):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "wanwuqianyi",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "strength": str(strength)
            }
        }
        
        print(f"[OAI Wanwuqianyi] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wanwuqianyi] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Wanwuqianyi] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Wanwuqianyi] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIKuotu:
    """OAI AI扩图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "top": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 600,
                    "step": 1
                }),
                "bottom": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 600,
                    "step": 1
                }),
                "left": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 600,
                    "step": 1
                }),
                "right": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 600,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, top, bottom, left, right):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Kuotu] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Kuotu] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Kuotu] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, top, bottom, left, right)
        print(f"[OAI Kuotu] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Kuotu] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Kuotu] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Kuotu] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Kuotu] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, top, bottom, left, right):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "kuotu",
            "parameter": {
                "image": image_url,
                "top": str(top),
                "bottom": str(bottom),
                "left": str(left),
                "right": str(right)
            }
        }
        
        print(f"[OAI Kuotu] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Kuotu] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Kuotu] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Kuotu] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIKoutu:
    """OAI AI抠图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Koutu] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Koutu] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Koutu] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Koutu] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Koutu] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Koutu] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Koutu] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Koutu] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "tupiankoutu",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Koutu] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Koutu] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Koutu] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Koutu] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIJianzhi:
    """OAI 万物可剪-新年剪纸风节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Jianzhi] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Jianzhi] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Jianzhi] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI Jianzhi] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Jianzhi] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Jianzhi] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Jianzhi] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Jianzhi] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "jianzhi",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI Jianzhi] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Jianzhi] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Jianzhi] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Jianzhi] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIShangse:
    """OAI AI线稿上色节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2, batch_size):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Shangse] 正在上传第一张图片到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Shangse] 第一张图片上传成功: {image1_url}")
        
        print(f"[OAI Shangse] 正在上传第二张图片到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Shangse] 第二张图片上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Shangse] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, batch_size)
        print(f"[OAI Shangse] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Shangse] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Shangse] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Shangse] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Shangse] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "xiangaoshangse",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Shangse] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Shangse] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Shangse] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Shangse] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Shangse] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Shangse] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIHuanyi:
    """OAI 一键换衣节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image1, image2):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI Huanyi] 正在上传第一张图片到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Huanyi] 第一张图片上传成功: {image1_url}")
        
        print(f"[OAI Huanyi] 正在上传第二张图片到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Huanyi] 第二张图片上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Huanyi] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url)
        print(f"[OAI Huanyi] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Huanyi] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Huanyi] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Huanyi] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Huanyi] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "yijianhuanyi",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url
            }
        }
        
        print(f"[OAI Huanyi] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Huanyi] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Huanyi] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Huanyi] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIWenshengshipin:
    """OAI 文生视频节点
    
    工作流程:
    1. 提交视频生成任务到API
    2. 轮询查询任务状态直到完成
    3. 获取视频URL后自动下载视频
    4. 使用OpenCV处理视频为帧序列
    5. 返回处理好的视频帧、帧数和视频信息
    """
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入视频描述提示词"
                }),
                "duration": (["5", "10"], {
                    "default": "5"
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {
                    "default": "9:16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, prompt, duration, aspect_ratio):
        """生成视频并处理为帧序列"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请填写视频描述")
        
        # 提交任务
        print(f"[OAI Wenshengshipin] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, duration, aspect_ratio)
        print(f"[OAI Wenshengshipin] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Wenshengshipin] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Wenshengshipin] 任务完成！")
                video_url = result["result"]
                print(f"[OAI Wenshengshipin] 视频URL: {video_url}")
                
                # 下载并处理视频
                return self._process_video_from_url(video_url)
                
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Wenshengshipin] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词不合规 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Wenshengshipin] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _submit_task(self, api_key, prompt, duration, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "wenshengshipin",
            "parameter": {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio
            }
        }
        
        print(f"[OAI Wenshengshipin] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wenshengshipin] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _process_video_from_url(self, url):
        """从URL下载并处理视频为帧序列"""
        print(f"[OAI Wenshengshipin] 开始下载视频: {url}")
        
        # 下载视频到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        print(f"[OAI Wenshengshipin] 视频下载完成，开始处理...")

        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(temp_file_path)
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"[OAI Wenshengshipin] 视频信息: {width}x{height}, {fps}fps, {total_frames}帧, 时长{duration:.2f}秒")

        # 处理视频帧 - 保持原始尺寸
        frames = []
        frame_count = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Numpy -> Torch张量，归一化到0-1
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame)
            frame_count += 1

        cap.release()
        
        # 清理临时文件
        os.unlink(temp_file_path)
        print(f"[OAI Wenshengshipin] 临时文件已删除")

        # 堆叠所有帧
        frames = torch.stack(frames)

        # 视频信息
        video_info = {
            "fps": fps,
            "frame_count": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
        }
        
        video_info_str = json.dumps(video_info, ensure_ascii=False, indent=2)
        print(f"[OAI Wenshengshipin] 处理完成! 共加载{frame_count}帧")
        print(f"[OAI Wenshengshipin] 视频信息:\n{video_info_str}")

        return (frames, frame_count, video_info_str)


class OAISora2:
    """OAI Sora2视频节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入视频描述提示词"
                }),
                "model": (["sora-2", "sora-2-pro"], {
                    "default": "sora-2"
                }),
                "hd": (["true", "false"], {
                    "default": "false"
                }),
                "duration": (["10", "15"], {
                    "default": "10"
                }),
                "duration2": (["10", "15", "25"], {
                    "default": "15"
                }),
                "aspect_ratio": (["16:9", "9:16"], {
                    "default": "16:9"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "可选：输入视频URL"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, prompt, model, hd, duration, duration2, aspect_ratio, image=None, video_url=None):
        """生成视频"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请填写视频描述")
        
        # 处理可选的图片上传
        images_url = None
        if image is not None:
            print(f"[OAI Sora2] 正在上传图片到OSS...")
            images_url = self._upload_image_to_oss(image)
            print(f"[OAI Sora2] 图片上传成功: {images_url}")
        
        # 提交任务
        print(f"[OAI Sora2] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, model, images_url, hd, duration, duration2, aspect_ratio, video_url)
        print(f"[OAI Sora2] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Sora2] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Sora2] 任务完成！")
                video_result_url = result["result"]
                print(f"[OAI Sora2] 视频URL: {video_result_url}")
                return process_video_from_url(video_result_url, "OAI Sora2")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Sora2] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词不合规 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Sora2] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, prompt, model, images_url, hd, duration, duration2, aspect_ratio, video_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建参数，只包含有值的字段
        parameter = {
            "prompt": prompt,
            "model": model,
            "hd": hd == "true",
            "duration": duration,
            "duration2": duration2,
            "aspect_ratio": aspect_ratio
        }
        
        # 添加可选参数
        if images_url:
            parameter["images"] = images_url
        
        if video_url and video_url.strip():
            parameter["video"] = video_url.strip()
        
        payload = {
            "appId": "soraapi",
            "parameter": parameter
        }
        
        print(f"[OAI Sora2] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Sora2] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIDongzuoqianyi:
    """OAI 视频动作迁移节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入视频URL"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请描述动作行为"
                }),
                "duration": (["1", "2", "3", "4", "5"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "transfer_motion"
    CATEGORY = "OAI"
    
    def transfer_motion(self, api_key, video_url, image, prompt, duration):
        """迁移视频动作"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not video_url or not video_url.strip():
            raise ValueError("视频URL不能为空，请填写视频URL")
        
        if not prompt or not prompt.strip():
            raise ValueError("动作描述不能为空，请描述动作行为")
        
        # 上传图片到OSS
        print(f"[OAI Dongzuoqianyi] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Dongzuoqianyi] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Dongzuoqianyi] 开始提交任务...")
        task_id = self._submit_task(api_key, video_url.strip(), image_url, prompt, duration)
        print(f"[OAI Dongzuoqianyi] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Dongzuoqianyi] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Dongzuoqianyi] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Dongzuoqianyi] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Dongzuoqianyi")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Dongzuoqianyi] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 视频URL无效 2) 图片格式不支持 3) API密钥权限不足 4) 账户余额不足 5) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Dongzuoqianyi] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, video_url, image_url, prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "shipindongzuoqianyi",
            "parameter": {
                "video": video_url,
                "image": image_url,
                "prompt": prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Dongzuoqianyi] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Dongzuoqianyi] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIDuikouxing:
    """OAI 视频对口型节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入视频URL"
                }),
                "audio_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入音频URL"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入描述词（可选）"
                }),
                "duration": (["1", "2", "3", "4", "5"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "lip_sync"
    CATEGORY = "OAI"
    
    def lip_sync(self, api_key, video_url, audio_url, prompt, duration):
        """视频对口型"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not video_url or not video_url.strip():
            raise ValueError("视频URL不能为空，请填写视频URL")
        
        if not audio_url or not audio_url.strip():
            raise ValueError("音频URL不能为空，请填写音频URL")
        
        # 提交任务
        print(f"[OAI Duikouxing] 开始提交任务...")
        task_id = self._submit_task(api_key, video_url.strip(), audio_url.strip(), prompt, duration)
        print(f"[OAI Duikouxing] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Duikouxing] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Duikouxing] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Duikouxing] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Duikouxing")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Duikouxing] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 视频URL无效 2) 音频URL无效 3) API密钥权限不足 4) 账户余额不足 5) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Duikouxing] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _submit_task(self, api_key, video_url, audio_url, prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "shipinduikouxing",
            "parameter": {
                "video": video_url,
                "audio": audio_url,
                "prompt": prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Duikouxing] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Duikouxing] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIShuziren:
    """OAI AI图片数字人节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "action_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入动作提示词"
                }),
                "duration": (["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "generate_digital_human"
    CATEGORY = "OAI"
    
    def generate_digital_human(self, api_key, image, audio, action_prompt, duration):
        """生成数字人视频"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not action_prompt or not action_prompt.strip():
            raise ValueError("动作提示词不能为空，请填写动作提示词")
        
        # 上传图片到OSS
        print(f"[OAI Shuziren] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Shuziren] 图片上传成功: {image_url}")
        
        # 上传音频到OSS
        print(f"[OAI Shuziren] 正在上传音频到OSS...")
        audio_url = self._upload_audio_to_oss(audio)
        print(f"[OAI Shuziren] 音频上传成功: {audio_url}")
        
        # 提交任务
        print(f"[OAI Shuziren] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, audio_url, action_prompt, duration)
        print(f"[OAI Shuziren] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Shuziren] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Shuziren] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Shuziren] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Shuziren")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Shuziren] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) 音频格式不支持 3) API密钥权限不足 4) 账户余额不足 5) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Shuziren] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _upload_audio_to_oss(self, audio):
        """将音频上传到OSS并返回URL"""
        try:
            # 调试信息：打印音频数据类型和结构
            print(f"[OAI Shuziren] 音频数据类型: {type(audio)}")
            
            audio_file_path = None
            temp_file_created = False
            
            # 处理不同格式的音频数据
            if isinstance(audio, dict):
                # 如果有文件路径，直接使用
                audio_file_path = audio.get('file_path') or audio.get('filename') or audio.get('path')
                
                # 如果没有文件路径，但有波形数据，需要转换为音频文件
                if not audio_file_path and 'waveform' in audio and 'sample_rate' in audio:
                    print(f"[OAI Shuziren] 检测到波形数据，准备转换为音频文件...")
                    import tempfile
                    import torchaudio
                    
                    waveform = audio['waveform']
                    sample_rate = audio['sample_rate']
                    
                    # 创建临时文件
                    temp_fd, audio_file_path = tempfile.mkstemp(suffix='.wav')
                    os.close(temp_fd)
                    temp_file_created = True
                    
                    print(f"[OAI Shuziren] 波形shape: {waveform.shape}, 采样率: {sample_rate}")
                    print(f"[OAI Shuziren] 保存到临时文件: {audio_file_path}")
                    
                    # 保存为WAV文件
                    torchaudio.save(audio_file_path, waveform.squeeze(0), sample_rate)
                    
            elif isinstance(audio, str):
                # 如果直接是字符串，可能就是文件路径
                audio_file_path = audio
            elif isinstance(audio, (list, tuple)) and len(audio) > 0:
                # 如果是列表或元组，取第一个元素
                first_item = audio[0]
                if isinstance(first_item, dict):
                    audio_file_path = first_item.get('file_path') or first_item.get('filename') or first_item.get('path')
                elif isinstance(first_item, str):
                    audio_file_path = first_item
            
            if not audio_file_path:
                raise ValueError(f"无法从音频数据中获取或生成文件路径，数据类型: {type(audio)}")
            
            # 检查文件是否存在
            if not os.path.exists(audio_file_path):
                raise ValueError(f"音频文件不存在: {audio_file_path}")
            
            print(f"[OAI Shuziren] 准备上传音频文件: {audio_file_path}")
            
            # 读取音频文件
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
            
            print(f"[OAI Shuziren] 音频文件大小: {len(audio_bytes)} bytes")
            
            # 获取文件扩展名
            file_ext = os.path.splitext(audio_file_path)[1] or '.wav'
            mime_type = 'audio/wav' if file_ext == '.wav' else ('audio/mpeg' if file_ext == '.mp3' else f'audio/{file_ext[1:]}')
            
            print(f"[OAI Shuziren] 文件扩展名: {file_ext}, MIME类型: {mime_type}")
            
            files = {'file': (f'audio{file_ext}', audio_bytes, mime_type)}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # 清理临时文件
            if temp_file_created and os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                    print(f"[OAI Shuziren] 已删除临时文件: {audio_file_path}")
                except:
                    pass
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    audio_url = data_field.get("url") or data_field.get("audioUrl") or data_field.get("fileUrl")
                    if audio_url:
                        return audio_url
                
                audio_url = data.get("url") or data.get("audioUrl") or data.get("fileUrl")
                if audio_url:
                    return audio_url
                
                raise Exception(f"OSS响应中未找到音频URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"音频上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, audio_url, action_prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "shuzirenkoubo",
            "parameter": {
                "image": image_url,
                "audio": audio_url,
                "action_prompt": action_prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Shuziren] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Shuziren] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAITihuanrenwu:
    """OAI AI视频替换人物节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入视频URL"
                }),
                "image": ("IMAGE",),
                "action_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入动作描述"
                }),
                "duration": (["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "replace_person"
    CATEGORY = "OAI"
    
    def replace_person(self, api_key, video_url, image, action_prompt, duration):
        """视频替换人物"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not video_url or not video_url.strip():
            raise ValueError("视频URL不能为空，请填写视频URL")
        
        if not action_prompt or not action_prompt.strip():
            raise ValueError("动作描述不能为空，请填写动作描述")
        
        # 上传图片到OSS
        print(f"[OAI Tihuanrenwu] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Tihuanrenwu] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Tihuanrenwu] 开始提交任务...")
        task_id = self._submit_task(api_key, video_url.strip(), image_url, action_prompt, duration)
        print(f"[OAI Tihuanrenwu] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Tihuanrenwu] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Tihuanrenwu] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Tihuanrenwu] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Tihuanrenwu")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Tihuanrenwu] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 视频URL无效 2) 图片格式不支持 3) API密钥权限不足 4) 账户余额不足 5) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Tihuanrenwu] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, video_url, image_url, action_prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "shipinrebwutihuan",
            "parameter": {
                "video": video_url,
                "image": image_url,
                "action_prompt": action_prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Tihuanrenwu] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Tihuanrenwu] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIShouweizhen:
    """OAI 首尾帧节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入描述词"
                }),
                "duration": (["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, image1, image2, prompt, duration):
        """首尾帧生成视频"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("描述词不能为空，请填写描述词")
        
        # 上传图片1到OSS
        print(f"[OAI Shouweizhen] 正在上传图片1到OSS...")
        image1_url = self._upload_image_to_oss(image1)
        print(f"[OAI Shouweizhen] 图片1上传成功: {image1_url}")
        
        # 上传图片2到OSS
        print(f"[OAI Shouweizhen] 正在上传图片2到OSS...")
        image2_url = self._upload_image_to_oss(image2)
        print(f"[OAI Shouweizhen] 图片2上传成功: {image2_url}")
        
        # 提交任务
        print(f"[OAI Shouweizhen] 开始提交任务...")
        task_id = self._submit_task(api_key, image1_url, image2_url, prompt, duration)
        print(f"[OAI Shouweizhen] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Shouweizhen] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Shouweizhen] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Shouweizhen] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Shouweizhen")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Shouweizhen] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Shouweizhen] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image1_url, image2_url, prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "shouweizhen",
            "parameter": {
                "image1": image1_url,
                "image2": image2_url,
                "prompt": prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Shouweizhen] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Shouweizhen] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAITushengshipin:
    """OAI 图生视频节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入描述词"
                }),
                "duration": (["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, image, prompt, duration):
        """图生视频"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("描述词不能为空，请填写描述词")
        
        # 上传图片到OSS
        print(f"[OAI Tushengshipin] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Tushengshipin] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Tushengshipin] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, prompt, duration)
        print(f"[OAI Tushengshipin] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Tushengshipin] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Tushengshipin] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Tushengshipin] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Tushengshipin")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Tushengshipin] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Tushengshipin] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "tushengshipin",
            "parameter": {
                "image": image_url,
                "prompt": prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Tushengshipin] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Tushengshipin] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIYinpinqudong:
    """OAI 音频驱动图片节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入描述词"
                }),
                "duration": (["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], {
                    "default": "1"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, image, audio, prompt, duration):
        """音频驱动图片生成视频"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("描述词不能为空，请填写描述词")
        
        # 上传图片到OSS
        print(f"[OAI Yinpinqudong] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Yinpinqudong] 图片上传成功: {image_url}")
        
        # 上传音频到OSS
        print(f"[OAI Yinpinqudong] 正在上传音频到OSS...")
        audio_url = self._upload_audio_to_oss(audio)
        print(f"[OAI Yinpinqudong] 音频上传成功: {audio_url}")
        
        # 提交任务
        print(f"[OAI Yinpinqudong] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, audio_url, prompt, duration)
        print(f"[OAI Yinpinqudong] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Yinpinqudong] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Yinpinqudong] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Yinpinqudong] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Yinpinqudong")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Yinpinqudong] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) 音频格式不支持 3) API密钥权限不足 4) 账户余额不足 5) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Yinpinqudong] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _upload_audio_to_oss(self, audio):
        """将音频上传到OSS并返回URL"""
        try:
            # 调试信息：打印音频数据类型和结构
            print(f"[OAI Yinpinqudong] 音频数据类型: {type(audio)}")
            
            audio_file_path = None
            temp_file_created = False
            
            # 处理不同格式的音频数据
            if isinstance(audio, dict):
                # 如果有文件路径，直接使用
                audio_file_path = audio.get('file_path') or audio.get('filename') or audio.get('path')
                
                # 如果没有文件路径，但有波形数据，需要转换为音频文件
                if not audio_file_path and 'waveform' in audio and 'sample_rate' in audio:
                    print(f"[OAI Yinpinqudong] 检测到波形数据，准备转换为音频文件...")
                    import tempfile
                    import torchaudio
                    
                    waveform = audio['waveform']
                    sample_rate = audio['sample_rate']
                    
                    # 创建临时文件
                    temp_fd, audio_file_path = tempfile.mkstemp(suffix='.wav')
                    os.close(temp_fd)
                    temp_file_created = True
                    
                    print(f"[OAI Yinpinqudong] 波形shape: {waveform.shape}, 采样率: {sample_rate}")
                    print(f"[OAI Yinpinqudong] 保存到临时文件: {audio_file_path}")
                    
                    # 保存为WAV文件
                    torchaudio.save(audio_file_path, waveform.squeeze(0), sample_rate)
                    
            elif isinstance(audio, str):
                # 如果直接是字符串，可能就是文件路径
                audio_file_path = audio
            elif isinstance(audio, (list, tuple)) and len(audio) > 0:
                # 如果是列表或元组，取第一个元素
                first_item = audio[0]
                if isinstance(first_item, dict):
                    audio_file_path = first_item.get('file_path') or first_item.get('filename') or first_item.get('path')
                elif isinstance(first_item, str):
                    audio_file_path = first_item
            
            if not audio_file_path:
                raise ValueError(f"无法从音频数据中获取或生成文件路径，数据类型: {type(audio)}")
            
            # 检查文件是否存在
            if not os.path.exists(audio_file_path):
                raise ValueError(f"音频文件不存在: {audio_file_path}")
            
            print(f"[OAI Yinpinqudong] 准备上传音频文件: {audio_file_path}")
            
            # 读取音频文件
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
            
            print(f"[OAI Yinpinqudong] 音频文件大小: {len(audio_bytes)} bytes")
            
            # 获取文件扩展名
            file_ext = os.path.splitext(audio_file_path)[1] or '.wav'
            mime_type = 'audio/wav' if file_ext == '.wav' else ('audio/mpeg' if file_ext == '.mp3' else f'audio/{file_ext[1:]}')
            
            print(f"[OAI Yinpinqudong] 文件扩展名: {file_ext}, MIME类型: {mime_type}")
            
            files = {'file': (f'audio{file_ext}', audio_bytes, mime_type)}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # 清理临时文件
            if temp_file_created and os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                    print(f"[OAI Yinpinqudong] 已删除临时文件: {audio_file_path}")
                except:
                    pass
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    audio_url = data_field.get("url") or data_field.get("audioUrl") or data_field.get("fileUrl")
                    if audio_url:
                        return audio_url
                
                audio_url = data.get("url") or data.get("audioUrl") or data.get("fileUrl")
                if audio_url:
                    return audio_url
                
                raise Exception(f"OSS响应中未找到音频URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"音频上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, audio_url, prompt, duration):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "pinpinqudongtupian",
            "parameter": {
                "image": image_url,
                "audio": audio_url,
                "prompt": prompt,
                "duration": duration
            }
        }
        
        print(f"[OAI Yinpinqudong] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Yinpinqudong] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIDuocanshipin:
    """OAI 多参视频节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "duration": (["5"], {
                    "default": "5"
                }),
                "video_aspect_ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9"
                }),
                "resolution": (["1080p"], {
                    "default": "1080p"
                }),
                "movement_amplitude": (["auto", "small", "medium", "large"], {
                    "default": "auto"
                }),
                "bgm": (["false", "true"], {
                    "default": "false"
                }),
                "model": (["viduq2", "viduq1"], {
                    "default": "viduq2"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, prompt, duration, video_aspect_ratio, resolution, movement_amplitude, bgm, model):
        """多参视频生成"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请填写提示词")
        
        # 转换bgm为布尔值
        bgm_bool = bgm == "true"
        
        # 提交任务
        print(f"[OAI Duocanshipin] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, duration, video_aspect_ratio, resolution, movement_amplitude, bgm_bool, model)
        print(f"[OAI Duocanshipin] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Duocanshipin] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Duocanshipin] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Duocanshipin] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Duocanshipin")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Duocanshipin] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词不合规 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Duocanshipin] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _submit_task(self, api_key, prompt, duration, video_aspect_ratio, resolution, movement_amplitude, bgm, model):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "duocanshipin",
            "parameter": {
                "prompt": prompt,
                "duration": duration,
                "video_aspect_ratio": video_aspect_ratio,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "bgm": bgm,
                "model": model
            }
        }
        
        print(f"[OAI Duocanshipin] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Duocanshipin] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIJimengshipin:
    """OAI 即梦视频3.0节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "mode": (["文生视频", "图生视频"], {
                    "default": "文生视频"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1", "3:2", "2:3"], {
                    "default": "16:9"
                }),
                "duration": (["5", "10"], {
                    "default": "5"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "OAI"
    
    def generate_video(self, api_key, mode, prompt, aspect_ratio, duration, image=None):
        """即梦视频3.0生成"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请填写提示词")
        
        # 图生视频模式必须提供图片
        if mode == "图生视频" and image is None:
            raise ValueError("图生视频模式必须提供图片输入")
        
        image_url = None
        if image is not None:
            # 上传图片到OSS
            print(f"[OAI Jimengshipin] 正在上传图片到OSS...")
            image_url = self._upload_image(api_key, image)
            print(f"[OAI Jimengshipin] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Jimengshipin] 开始提交任务...")
        task_id = self._submit_task(api_key, mode, prompt, aspect_ratio, duration, image_url)
        print(f"[OAI Jimengshipin] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI Jimengshipin] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Jimengshipin] 任务完成！")
                result_video_url = result["result"]
                print(f"[OAI Jimengshipin] 视频URL: {result_video_url}")
                return process_video_from_url(result_video_url, "OAI Jimengshipin")
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Jimengshipin] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词不合规 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Jimengshipin] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image(self, api_key, image):
        """上传图片到OSS"""
        # 将tensor转换为PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_array = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        
        # 转换为字节流
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # 上传到OSS
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        files = {
            'file': ('image.png', img_byte_arr, 'image/png')
        }
        
        try:
            response = requests.post(self.upload_url, headers=headers, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Jimengshipin] OSS上传响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"图片上传失败 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data or "url" not in data["data"]:
                raise Exception(f"图片上传返回格式错误: {data}")
            
            return data["data"]["url"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, mode, prompt, aspect_ratio, duration, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "jimengshipin",
            "parameter": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "text": mode
            }
        }
        
        # 如果有图片URL，添加到参数中
        if image_url:
            payload["parameter"]["image"] = image_url
        
        print(f"[OAI Jimengshipin] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Jimengshipin] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class OAIDoubao40:
    """OAI 豆包绘图4.0节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "model": (["img2imgs", "text2imgs"], {
                    "default": "img2imgs"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, prompt, model, aspect_ratio, seed, image1=None, image2=None, image3=None, image4=None):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空，请输入提示词")
        
        # 检查模式与图片的匹配
        has_images = any([image1 is not None, image2 is not None, image3 is not None, image4 is not None])
        if model == "img2imgs" and not has_images:
            raise ValueError("选择img2imgs模式时，至少需要提供一张图片")
        
        # 收集图片并上传到OSS
        image_urls = {}
        if has_images:
            for i, img in enumerate([image1, image2, image3, image4], 1):
                if img is not None:
                    print(f"[OAI Doubao4.0] 正在上传图片{i}到OSS...")
                    url = self._upload_image_to_oss(img)
                    image_urls[f"image{i}"] = url
                    print(f"[OAI Doubao4.0] 图片{i}上传成功: {url}")
        
        # 提交任务
        print(f"[OAI Doubao4.0] 开始提交任务...")
        task_id = self._submit_task(api_key, image_urls, prompt, model, aspect_ratio)
        print(f"[OAI Doubao4.0] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Doubao4.0] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Doubao4.0] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Doubao4.0] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词包含敏感内容被审核拒绝 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI Doubao4.0] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_urls, prompt, model, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建 parameter
        parameter = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio
        }
        # 添加图片URL
        parameter.update(image_urls)
        
        payload = {
            "appId": "doubao4.0",
            "parameter": parameter
        }
        
        print(f"[OAI Doubao4.0] 提交参数: appId={payload['appId']}, model={model}, aspect_ratio={aspect_ratio}, images={list(image_urls.keys())}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Doubao4.0] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Doubao4.0] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Doubao4.0] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIWanwuhuanbeijing:
    """OAI 万物换背景节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/uploadNodeFile"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入背景描述"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "OAI"
    
    def process_image(self, api_key, image, prompt, aspect_ratio):
        """处理图像"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        if not prompt or not prompt.strip():
            raise ValueError("背景描述不能为空，请输入背景描述")
        
        # 将图片上传到OSS获取URL
        print(f"[OAI Wanwuhuanbeijing] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI Wanwuhuanbeijing] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI Wanwuhuanbeijing] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url, prompt, aspect_ratio)
        print(f"[OAI Wanwuhuanbeijing] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Wanwuhuanbeijing] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Wanwuhuanbeijing] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                # 尝试获取更详细的错误信息
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Wanwuhuanbeijing] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                # 如果没有详细错误信息，给出可能的原因提示
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词包含敏感内容被审核拒绝 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                # 任务仍在处理中
                print(f"[OAI Wanwuhuanbeijing] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            # 将tensor转换为PIL Image
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            # 转换为bytes
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            # 上传到OSS
            files = {'file': ('image.png', img_bytes, 'image/png')}
            
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wanwuhuanbeijing] OSS响应: {json.dumps(data, ensure_ascii=False)}")
            
            # 根据常见的OSS响应格式获取URL
            if data.get("code") == 200 or data.get("success"):
                # data字段可能直接是字符串URL，也可能是对象
                data_field = data.get("data")
                if isinstance(data_field, str):
                    # data直接是URL字符串
                    return data_field
                elif isinstance(data_field, dict):
                    # data是对象，尝试获取url字段
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                # 尝试其他可能的字段
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url, prompt, aspect_ratio):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "58307ba6-9114-4587-bfcb-1f0b0293ee0a",
            "parameter": {
                "image": image_url,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio
            }
        }
        
        print(f"[OAI Wanwuhuanbeijing] 提交参数: appId={payload['appId']}, image_url={image_url}, prompt={prompt}, aspect_ratio={aspect_ratio}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            # 打印响应状态码和内容以便调试
            print(f"[OAI Wanwuhuanbeijing] 响应状态码: {response.status_code}")
            if response.status_code != 200:
                print(f"[OAI Wanwuhuanbeijing] 错误响应内容: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wanwuhuanbeijing] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            # 检查返回数据结构
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            # API返回的是taskId（驼峰命名）或task_id
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 使用GET方法，taskId作为路径参数
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wanwuhuanbeijing] 查询响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            print(f"[OAI Wanwuhuanbeijing] 正在下载图像: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            # 将图像数据转换为PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # 转换为RGB模式（如果是RGBA或其他模式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为torch tensor，格式为 [batch, height, width, channels]
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[OAI Wanwuhuanbeijing] 图像下载完成，尺寸: {image.size}")
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIWanTextToImage:
    """OAI Wan2.2文生图节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "请输入提示词"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "9:16"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "OAI"
    
    def generate_image(self, api_key, prompt, aspect_ratio, batch_size, seed):
        """生成图像"""
        
        print(f"[OAI Wan2.2] ========== 节点开始执行 ==========")
        print(f"[OAI Wan2.2] 接收到的参数:")
        print(f"[OAI Wan2.2]   - api_key: {'已提供' if api_key and api_key.strip() else '未提供或为空'}")
        print(f"[OAI Wan2.2]   - prompt: '{prompt}'")
        print(f"[OAI Wan2.2]   - aspect_ratio: {aspect_ratio}")
        print(f"[OAI Wan2.2]   - batch_size: {batch_size}")
        print(f"[OAI Wan2.2]   - seed: {seed}")
        
        if not api_key or not api_key.strip():
            error_msg = "API密钥不能为空，请填写您的API密钥"
            print(f"[OAI Wan2.2] 错误: {error_msg}")
            raise ValueError(error_msg)
        
        if not prompt or not prompt.strip():
            error_msg = "提示词不能为空，请输入提示词"
            print(f"[OAI Wan2.2] 错误: {error_msg}")
            raise ValueError(error_msg)
        
        # 提交任务
        print(f"[OAI Wan2.2] 开始提交任务...")
        task_id = self._submit_task(api_key, prompt, aspect_ratio, batch_size)
        print(f"[OAI Wan2.2] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每5秒检查一次，最多10分钟（120次）
        max_attempts = 120
        check_interval = 5
        
        for attempt in range(max_attempts):
            print(f"[OAI Wan2.2] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI Wan2.2] 任务完成！")
                image_url = result["result"]
                return self._download_image(image_url)
            elif result["status"] == "failed":
                # 尝试获取更详细的错误信息
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI Wan2.2] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                # 如果没有详细错误信息，给出可能的原因提示
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 提示词包含敏感内容被审核拒绝 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                # 任务仍在处理中
                print(f"[OAI Wan2.2] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过10分钟），任务ID: {task_id}")
    
    def _submit_task(self, api_key, prompt, aspect_ratio, batch_size):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "wan2.2wenshengtu",
            "parameter": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "batch_size": str(batch_size)
            }
        }
        
        print(f"[OAI Wan2.2] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wan2.2] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            # 检查返回数据结构
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            # API返回的是taskId（驼峰命名）
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 使用GET方法，taskId作为路径参数
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI Wan2.2] 查询响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")
    
    def _download_image(self, image_url):
        """下载图像并转换为ComfyUI格式"""
        try:
            # 处理多个URL的情况（用逗号分隔）
            image_urls = [url.strip() for url in image_url.split(',') if url.strip()]
            
            print(f"[OAI Wan2.2] 正在下载 {len(image_urls)} 张图像...")
            
            image_tensors = []
            for idx, url in enumerate(image_urls, 1):
                print(f"[OAI Wan2.2] 下载第 {idx}/{len(image_urls)} 张: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                # 将图像数据转换为PIL Image
                image = Image.open(io.BytesIO(response.content))
                
                # 转换为RGB模式（如果是RGBA或其他模式）
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 转换为numpy数组
                image_np = np.array(image).astype(np.float32) / 255.0
                
                # 转换为torch tensor
                image_tensor = torch.from_numpy(image_np)
                image_tensors.append(image_tensor)
                
                print(f"[OAI Wan2.2] 第 {idx} 张图像下载完成，尺寸: {image.size}")
            
            # 将所有图像堆叠成batch，格式为 [batch, height, width, channels]
            result_tensor = torch.stack(image_tensors)
            print(f"[OAI Wan2.2] 所有图像下载完成，总共 {len(image_tensors)} 张")
            
            return (result_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"下载图像失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理图像失败: {str(e)}")


class OAIImagePromptReverse:
    """OAI 图像提示词反推节点"""
    
    def __init__(self):
        self.api_url = "https://oaigc.cn/api/v1/task/submit"
        self.query_url = "https://oaigc.cn/api/v1/task/query"
        self.upload_url = "https://oaigc.cn/api/file/tool/upload"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的API密钥"
                }),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "OAI"
    
    def reverse_prompt(self, api_key, image):
        """反推图像提示词"""
        
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空，请填写您的API密钥")
        
        # 上传图片到OSS
        print(f"[OAI ImagePromptReverse] 正在上传图片到OSS...")
        image_url = self._upload_image_to_oss(image)
        print(f"[OAI ImagePromptReverse] 图片上传成功: {image_url}")
        
        # 提交任务
        print(f"[OAI ImagePromptReverse] 开始提交任务...")
        task_id = self._submit_task(api_key, image_url)
        print(f"[OAI ImagePromptReverse] 任务已提交，任务ID: {task_id}")
        
        # 轮询查询任务结果，每10秒检查一次，最多30分钟（180次）
        max_attempts = 180
        check_interval = 10
        
        for attempt in range(max_attempts):
            print(f"[OAI ImagePromptReverse] 检查任务状态 ({attempt + 1}/{max_attempts})...")
            time.sleep(check_interval)
            
            result = self._query_task(api_key, task_id)
            
            if result["status"] == "success":
                print(f"[OAI ImagePromptReverse] 任务完成！")
                prompt_result = result["result"]
                print(f"[OAI ImagePromptReverse] 反推提示词: {prompt_result}")
                return (prompt_result,)
            elif result["status"] == "failed":
                error_msg = result.get('error') or result.get('message') or result.get('error_message') or result.get('fail_reason') or '未知错误'
                print(f"[OAI ImagePromptReverse] 任务失败详情: {json.dumps(result, ensure_ascii=False)}")
                
                if error_msg == '未知错误':
                    error_msg = '任务失败，可能原因：1) 图片格式不支持 2) API密钥权限不足 3) 账户余额不足 4) 服务端错误'
                
                raise Exception(f"任务失败: {error_msg}")
            else:
                print(f"[OAI ImagePromptReverse] 任务处理中，状态: {result['status']}")
        
        raise TimeoutError(f"任务超时（超过30分钟），任务ID: {task_id}")
    
    def _upload_image_to_oss(self, image_tensor):
        """将图片上传到OSS并返回URL"""
        try:
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_bytes = buffered.getvalue()
            
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(self.upload_url, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == 200 or data.get("success"):
                data_field = data.get("data")
                if isinstance(data_field, str):
                    return data_field
                elif isinstance(data_field, dict):
                    image_url = data_field.get("url") or data_field.get("imageUrl")
                    if image_url:
                        return image_url
                
                image_url = data.get("url") or data.get("imageUrl")
                if image_url:
                    return image_url
                
                raise Exception(f"OSS响应中未找到图片URL: {data}")
            else:
                raise Exception(f"OSS上传失败: {data.get('message', '未知错误')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OSS上传失败: {str(e)}")
        except Exception as e:
            raise Exception(f"图片上传失败: {str(e)}")
    
    def _submit_task(self, api_key, image_url):
        """提交任务"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "appId": "tuxiangfantui",
            "parameter": {
                "image": image_url
            }
        }
        
        print(f"[OAI ImagePromptReverse] 提交参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"[OAI ImagePromptReverse] API响应: {json.dumps(data, ensure_ascii=False)}")
            
            if data.get("code") != 200:
                raise Exception(f"API返回错误 (code: {data.get('code')}): {data.get('message', '未知错误')}")
            
            if "data" not in data:
                raise Exception(f"API返回数据格式错误，缺少data字段: {data}")
            
            if "task_id" not in data["data"] and "taskId" not in data["data"]:
                raise Exception(f"API返回数据格式错误，缺少taskId字段: {data['data']}")
            
            task_id = data["data"].get("taskId") or data["data"].get("task_id")
            return task_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"提交任务失败: {str(e)}")
    
    def _query_task(self, api_key, task_id):
        """查询任务状态"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.query_url}/{task_id}", headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") != 200:
                raise Exception(f"查询任务失败: {data.get('message', '未知错误')}")
            
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {str(e)}")


class LoadVideoFromURL:
    """从URL加载视频节点
    
    工作流程详解:
    1. 下载视频 (Download the video)
       - 获取视频URL地址
       - 创建临时MP4文件
       - 通过requests.get()流式下载视频数据到临时文件
    
    2. 读取视频 (Load the video)
       - 使用OpenCV(cv2)打开临时视频文件
       - 获取视频属性: fps(帧率), total_frames(总帧数), width(宽度), height(高度)
    
    3. 处理视频帧 (Process the frames)
       - skip_first_frames: 跳过开头指定数量的帧
       - select_every_nth: 每隔N帧选取一帧(抽帧功能)
       - force_size: 调整帧的尺寸
       - frame_load_cap: 限制加载的帧数上限
    
    4. 转换格式 (Format Conversion)
       - BGR转RGB颜色空间
       - Numpy数组转Torch张量
       - 像素值归一化到0-1范围
    
    5. 输出结果 (Return the results)
       - frames: 处理后的视频帧张量(IMAGE格式)
       - frame_count: 实际加载的帧数
       - video_info: 视频详细信息字典
    
    6. 清理工作 (Clean up)
       - 删除临时视频文件
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/video.mp4"}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                "custom_width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "load_video_from_url"
    CATEGORY = "OAI"

    def load_video_from_url(self, url, force_rate, force_size, custom_width, custom_height, frame_load_cap, skip_first_frames, select_every_nth):
        """从URL加载视频"""
        print(f"[LoadVideoFromURL] 开始从URL加载视频: {url}")
        
        # 1. 下载视频到临时文件 (Download the video)
        print(f"[LoadVideoFromURL] 正在下载视频...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 流式写入数据
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        print(f"[LoadVideoFromURL] 视频下载完成: {temp_file_path}")

        # 2. 使用OpenCV读取视频 (Load the video)
        cap = cv2.VideoCapture(temp_file_path)
        
        # 获取视频属性 (Get video properties)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"[LoadVideoFromURL] 视频信息: {width}x{height}, {fps}fps, {total_frames}帧, 时长{duration:.2f}秒")

        # 3. 计算目标尺寸 (Calculate target size)
        if force_size != "Disabled":
            if force_size == "Custom Width":
                new_height = int(height * (custom_width / width))
                new_width = custom_width
            elif force_size == "Custom Height":
                new_width = int(width * (custom_height / height))
                new_height = custom_height
            elif force_size == "Custom":
                new_width, new_height = custom_width, custom_height
            else:
                # 解析类似 "512x?" 或 "?x512" 的格式
                target_width, target_height = map(int, force_size.replace("?", "0").split("x"))
                if target_width == 0:
                    new_width = int(width * (target_height / height))
                    new_height = target_height
                else:
                    new_height = int(height * (target_width / width))
                    new_width = target_width
        else:
            new_width, new_height = width, height

        print(f"[LoadVideoFromURL] 目标尺寸: {new_width}x{new_height}")

        # 4. 处理视频帧 (Process the frames)
        frames = []
        frame_count = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 跳过开头的帧 (skip_first_frames)
            if i < skip_first_frames:
                continue

            # 每隔N帧选一帧 (select_every_nth)
            if (i - skip_first_frames) % select_every_nth != 0:
                continue

            # 调整尺寸 (force_size)
            if force_size != "Disabled":
                frame = cv2.resize(frame, (new_width, new_height))

            # 5. 转换格式 (Format Conversion)
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Numpy -> Torch张量，归一化到0-1
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame)

            frame_count += 1

            # 达到加载上限 (frame_load_cap)
            if frame_load_cap > 0 and frame_count >= frame_load_cap:
                break

        cap.release()
        
        # 6. 清理临时文件 (Clean up)
        os.unlink(temp_file_path)
        print(f"[LoadVideoFromURL] 临时文件已删除")

        # 堆叠所有帧
        frames = torch.stack(frames)

        # 视频信息
        video_info = {
            "source_fps": fps,
            "source_frame_count": total_frames,
            "source_duration": duration,
            "source_width": width,
            "source_height": height,
            "loaded_fps": fps if force_rate == 0 else force_rate,
            "loaded_frame_count": frame_count,
            "loaded_duration": frame_count / (fps if force_rate == 0 else force_rate) if fps > 0 else 0,
            "loaded_width": new_width,
            "loaded_height": new_height,
        }
        
        video_info_str = json.dumps(video_info, ensure_ascii=False, indent=2)
        print(f"[LoadVideoFromURL] 加载完成! 共加载{frame_count}帧")
        print(f"[LoadVideoFromURL] 视频信息:\n{video_info_str}")

        return (frames, frame_count, video_info_str)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "OAIQwenTextToImage": OAIQwenTextToImage,
    "OAIJimengTextToImage": OAIJimengTextToImage,
    "OAIWanwurongtu": OAIWanwurongtu,
    "OAIZhengJianZhao": OAIZhengJianZhao,
    "OAIRenwuchangjingronghe": OAIRenwuchangjingronghe,
    "OAIZitaiqianyi": OAIZitaiqianyi,
    "OAIQianzibaitai": OAIQianzibaitai,
    "OAIQwenEdit": OAIQwenEdit,
    "OAIFantuichutu": OAIFantuichutu,
    "OAIXiangaochengtu": OAIXiangaochengtu,
    "OAIChongdaguang": OAIChongdaguang,
    "OAIZhenrenshouban": OAIZhenrenshouban,
    "OAIQushuiyin": OAIQushuiyin,
    "OAILaozhaopian": OAILaozhaopian,
    "OAIDianshang": OAIDianshang,
    "OAIFlux": OAIFlux,
    "OAIKeling": OAIKeling,
    "OAIJipuli": OAIJipuli,
    "OAIChanpin": OAIChanpin,
    "OAIGaoqing": OAIGaoqing,
    "OAIMaopei": OAIMaopei,
    "OAIWanwuqianyi": OAIWanwuqianyi,
    "OAIKuotu": OAIKuotu,
    "OAIKoutu": OAIKoutu,
    "OAIJianzhi": OAIJianzhi,
    "OAIShangse": OAIShangse,
    "OAIHuanyi": OAIHuanyi,
    "OAIWenshengshipin": OAIWenshengshipin,
    "OAISora2": OAISora2,
    "OAIDongzuoqianyi": OAIDongzuoqianyi,
    "OAIDuikouxing": OAIDuikouxing,
    "OAIShuziren": OAIShuziren,
    "OAITihuanrenwu": OAITihuanrenwu,
    "OAIShouweizhen": OAIShouweizhen,
    "OAITushengshipin": OAITushengshipin,
    "OAIYinpinqudong": OAIYinpinqudong,
    "OAIDuocanshipin": OAIDuocanshipin,
    "OAIJimengshipin": OAIJimengshipin,
    "OAIDoubao40": OAIDoubao40,
    "OAIWanwuhuanbeijing": OAIWanwuhuanbeijing,
    "OAIWanTextToImage": OAIWanTextToImage,
    "OAIImagePromptReverse": OAIImagePromptReverse,
    "LoadVideoFromURL": LoadVideoFromURL
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "OAIQwenTextToImage": "OAI-千问文生图",
    "OAIJimengTextToImage": "OAI-即梦绘画",
    "OAIWanwurongtu": "OAI-万物溶图",
    "OAIZhengJianZhao": "OAI-AI证件照",
    "OAIRenwuchangjingronghe": "OAI-人物场景融合",
    "OAIZitaiqianyi": "OAI-人物姿态迁移",
    "OAIQianzibaitai": "OAI-千姿百态",
    "OAIQwenEdit": "OAI-Qwen编辑图像",
    "OAIFantuichutu": "OAI-AI反推出图",
    "OAIXiangaochengtu": "OAI-线稿成图",
    "OAIChongdaguang": "OAI-AI废片拯救",
    "OAIZhenrenshouban": "OAI-真人一键手办",
    "OAIQushuiyin": "OAI-智能消除",
    "OAILaozhaopian": "OAI-老照片修复",
    "OAIDianshang": "OAI-电商带货",
    "OAIFlux": "OAI-Flux图像编辑",
    "OAIKeling": "OAI-可灵绘图",
    "OAIJipuli": "OAI-吉卜力风格",
    "OAIChanpin": "OAI-产品重打光",
    "OAIGaoqing": "OAI-高清放大",
    "OAIMaopei": "OAI-毛胚房装修",
    "OAIWanwuqianyi": "OAI-万物迁移",
    "OAIKuotu": "OAI-AI扩图",
    "OAIKoutu": "OAI-AI抠图",
    "OAIJianzhi": "OAI-新年剪纸风",
    "OAIShangse": "OAI-AI线稿上色",
    "OAIHuanyi": "OAI-一键换衣",
    "OAIWenshengshipin": "OAI-文生视频",
    "OAISora2": "OAI-Sora2视频",
    "OAIDongzuoqianyi": "OAI-视频动作迁移",
    "OAIDuikouxing": "OAI-视频对口型",
    "OAIShuziren": "OAI-AI图片数字人",
    "OAITihuanrenwu": "OAI-AI视频替换人物",
    "OAIShouweizhen": "OAI-首尾帧",
    "OAITushengshipin": "OAI-图生视频",
    "OAIYinpinqudong": "OAI-音频驱动图片",
    "OAIDuocanshipin": "OAI-多参视频",
    "OAIJimengshipin": "OAI-即梦视频3.0",
    "OAIDoubao40": "OAI-doubao4.0",
    "OAIWanwuhuanbeijing": "OAI-万物换背景",
    "OAIWanTextToImage": "OAI-Wan2.2文生图",
    "OAIImagePromptReverse": "OAI-图像提示词反推",
    "LoadVideoFromURL": "OAI-加载视频URL"
}
