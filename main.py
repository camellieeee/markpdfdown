import os
import sys
import time
import shutil
import logging
from core import LLMClient
from core.FileWorker import create_worker
from core.Util import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

def completion(message, model="", system_prompt="", image_paths=None, temperature=0.5, max_tokens=8192, retry_times=3):
    """
    Call OpenAI's completion interface for text generation

    Args:
        message (str): User input message
        model (str): Model name
        system_prompt (str, optional): System prompt, defaults to empty string
        image_paths (List[str], optional): List of image paths, defaults to None
        temperature (float, optional): Temperature for text generation, defaults to 0.5
        max_tokens (int, optional): Maximum number of tokens for generated text, defaults to 8192
    Returns:
        str: Generated text content
    """
    
    # Get API key and API base URL from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Please set the OPENAI_API_KEY environment variables")
        exit(1)
    
    # 获取API基础URL
    base_url = os.getenv("OPENAI_API_BASE")
    # 确保base_url不包含models路径
    if base_url and '/models' in base_url:
        base_url = base_url.split('/models')[0]
    
    if not base_url:
        base_url = "https://api.openai.com"
        
    logger.info(f"Using API base URL: {base_url}")
    
    # If no model is specified, use the default model
    if not model:
        model = os.getenv("OPENAI_DEFAULT_MODEL")
        if not model:
            model = "gpt-4o"

    # Initialize LLMClient
    client = LLMClient.LLMClient(base_url=base_url, api_key=api_key, model=model)
    # Call completion method with retry mechanism
    for _ in range(retry_times):
        try:
            response = client.completion(user_message=message, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature, max_tokens=max_tokens)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            # If retry fails, wait for a while before retrying
            time.sleep(0.5)
    return ""

def convert_image_to_markdown(image_path):
    """
    Convert image to Markdown format
    Args:
        image_path (str): Path to the image
    Returns:
        str: Converted Markdown string
    """
    user_prompt = """
Please read the content in the image and transcribe it into plain Markdown format. Please note:
1. Maintain the format of headings, text, formulas, and table rows and columns
2. Mathematical formulas should be transcribed using LaTeX syntax, ensuring consistency with the original
3. No additional explanation is needed, and no content outside the original text should be added.
    """
    
    response = completion(message=user_prompt, model="", image_paths=[image_path], temperature=0.3, max_tokens=8192)
    response = remove_markdown_warp(response, "markdown")
    return response

if __name__ == "__main__":
    # 检查命令行参数
    help_msg = "Usage: python main.py [input.pdf] [output.md] [start_page] [end_page]\n" \
               "   or: python main.py [start_page] [end_page] < input.pdf"
    
    input_path = None
    output_path = None
    start_page = 1
    end_page = 0
    
    # 解析命令行参数
    if len(sys.argv) >= 2:
        # 尝试判断第一个参数是文件路径还是页码
        if os.path.isfile(sys.argv[1]) or sys.argv[1].endswith('.pdf'):
            # 文件路径模式
            input_path = sys.argv[1]
            
            if len(sys.argv) >= 3:
                # 检查第二个参数是输出文件还是页码
                if sys.argv[2].endswith('.md'):
                    output_path = sys.argv[2]
                    if len(sys.argv) >= 4:
                        start_page = int(sys.argv[3])
                    if len(sys.argv) >= 5:
                        end_page = int(sys.argv[4])
                else:
                    # 第二个参数是起始页码
                    start_page = int(sys.argv[2])
                    if len(sys.argv) >= 4:
                        end_page = int(sys.argv[3])
        else:
            # 标准输入模式，参数是页码
            try:
                start_page = int(sys.argv[1])
                if len(sys.argv) >= 3:
                    end_page = int(sys.argv[2])
            except ValueError:
                logger.error("Invalid page number")
                logger.error(help_msg)
                exit(1)
    
    # 处理输入
    input_data = None
    if input_path:
        # 从文件读取
        with open(input_path, 'rb') as f:
            input_data = f.read()
            input_filename = os.path.basename(input_path)
    else:
        # 从标准输入读取
        input_data = sys.stdin.buffer.read()
        if not input_data:
            logger.error("No input data received")
            logger.error(help_msg)
            exit(1)
        input_filename = os.path.basename(sys.stdin.buffer.name)

    # Create output directory
    output_dir = f"output/{time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Try to get extension from file name
    input_ext = os.path.splitext(input_filename)[1]
    
    # If there is no extension or the file comes from standard input, try to determine the type by file content
    if not input_ext or input_filename == '<stdin>':
        # PDF file magic number/signature is %PDF-
        if input_data.startswith(b'%PDF-'):
            input_ext = '.pdf'
            logger.info("Recognized as PDF file by file content")
        # JPEG file magic number/signature is FF D8 FF DB
        elif input_data.startswith(b'\xFF\xD8\xFF\xDB'):
            input_ext = '.jpg'
            logger.info("Recognized as JPEG file by file content")
        # PNG file magic number/signature is 89 50 4E 47
        elif input_data.startswith(b'\x89\x50\x4E\x47'):
            input_ext = '.png'
            logger.info("Recognized as PNG file by file content")
        # BMP file magic number/signature is 42 4D
        elif input_data.startswith(b'\x42\x4D'):
            input_ext = '.bmp'
            logger.info("Recognized as BMP file by file content")
        else:
            logger.error("Unsupported file type")
            exit(1)
    
    temp_input_path = os.path.join(output_dir, f"input{input_ext}")
    with open(temp_input_path, "wb") as f:
        f.write(input_data)

    # create file worker
    try:
        worker = create_worker(temp_input_path, start_page, end_page)
    except ValueError as e:
        logger.error(str(e))
        exit(1)
    
    # convert to images
    img_paths = worker.convert_to_images()
    logger.info("Image conversion completed")

    # convert to markdown
    markdown = ""
    for img_path in sorted(img_paths):
        img_path = img_path.replace("\\", "/")
        logger.info("Converting image %s to Markdown", img_path)
        markdown += convert_image_to_markdown(img_path)
        markdown += "\n\n"
    logger.info("Image conversion to Markdown completed")
    
    # Output Markdown
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        logger.info(f"Markdown saved to {output_path}")
    else:
        print(markdown)
    
    # Remove output path
    shutil.rmtree(output_dir)
    exit(0)
    