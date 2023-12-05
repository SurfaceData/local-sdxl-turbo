import base64
import torch
import uvicorn

from argparse import ArgumentParser
from diffusers import AutoPipelineForText2Image
from io import BytesIO
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()


# Mimic the OpenAI image generate request format with defaults matching what
# this server supports.
class CreateImageRequest(BaseModel):
    prompt: str
    model: str = "sdxl-turbo"
    n: int = 1
    quality: Union[str, None] = "standard"
    response_format: Union[str, None] = "b64_json"
    size: Union[str, None] = "512x512"
    style: Union[str, None] = "natural"
    user: Union[str, None] = None


# Mimic the OpenAI image result object.
class ImageResult(BaseModel):
    url: Union[str, None] = None
    b64_json: Union[str, None] = None
    revised_prompt: str


# Mimic the OpenAI image response format.
class CreateImageResponse(BaseModel):
    created: int
    data: List[ImageResult]


# Generate an image given a prompt.
@app.post("/v1/images/generations")
def imagesGenerations(request: CreateImageRequest) -> CreateImageResponse:
    image = app.pipe(
        prompt=request.prompt, num_inference_steps=1, guidance_scale=0.0
    ).images[0]
    # Right now, only support base64 encoded results.  Let clients figure out
    # how to save the images.
    buffer = BytesIO()
    image.save(buffer, format="png")
    image_b64 = base64.b64encode(buffer.getvalue())
    return CreateImageResponse(
        created=1589478378,
        data=[ImageResult(b64_json=image_b64, revised_prompt=request.prompt)],
    )


if __name__ == "__main__":
    # Handle some standard arguments
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Load up the model on the appropriate device.
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(args.device)
    app.pipe = pipe

    # Serve!
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
