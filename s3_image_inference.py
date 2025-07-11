import boto3
import base64
import io
from openai import OpenAI
from PIL import Image

def load_image_from_s3(s3_url):
    """Load image from S3 URL and return PIL Image object"""
    from urllib.parse import urlparse
    
    # Parse S3 URL: s3://bucket-name/path/to/file.jpg
    parsed = urlparse(s3_url)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip('/')
    
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    image_data = response['Body'].read()
    return Image.open(io.BytesIO(image_data))

def encode_image_to_base64(img, format="JPEG"):
    """Convert PIL image to base64 string"""
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode("utf-8")

def run_openai_inference(base64_image, prompt="What's in this image?", model="gpt-4o"):
    """Run OpenAI inference on base64 encoded image"""
    client = OpenAI()
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content

# Main function - put it all together
def process_s3_image(s3_url, prompt="What's in this image?"):
    """Complete pipeline: S3 → PIL → Base64 → OpenAI inference"""
    try:
        # 1. Load from S3
        img = load_image_from_s3(s3_url)
        print(f"✅ Loaded image from {s3_url}")
        
        # 2. Encode to base64
        base64_image = encode_image_to_base64(img)
        print(f"✅ Encoded image to base64 ({len(base64_image)} chars)")
        
        # 3. Run OpenAI inference
        result = run_openai_inference(base64_image, prompt)
        print(f"🤖 OpenAI response: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your S3 URL
    s3_url = "s3://your-bucket-name/path/to/your/image.jpg"
    
    result = process_s3_image(s3_url, "Describe this image in detail") 