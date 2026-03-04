# ColQwen2.5 Embedding Service

RunPod serverless endpoint serving [ColQwen2.5](https://huggingface.co/vidore/colqwen2.5-v0.1) multi-vector embeddings (128-dim per patch/token).

## Operations

All requests go through RunPod's API:

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
Authorization: Bearer {RUNPOD_API_KEY}
Content-Type: application/json
```

### Embed text query

```json
{
  "input": {
    "operation": "embed_query",
    "query": "invoice total amount",
    "api_key": "your-embedding-api-key"
  }
}
```

Response: `{"output": {"embeddings": [[float, ...], ...]}}` — shape `[num_tokens, 128]`

### Embed single image

```json
{
  "input": {
    "operation": "embed_image",
    "image_b64": "<base64-encoded PNG>",
    "api_key": "your-embedding-api-key"
  }
}
```

Response: `{"output": {"embeddings": [[float, ...], ...]}}` — shape `[num_patches, 128]`

### Embed batch of images

```json
{
  "input": {
    "operation": "embed_images",
    "images_b64": ["<base64 PNG>", "<base64 PNG>", ...],
    "api_key": "your-embedding-api-key"
  }
}
```

Response: `{"output": {"embeddings": [[[float, ...], ...], ...]}}` — list of `[num_patches, 128]`

## Deployment

### 1. Build and push Docker image

```bash
docker build --platform linux/amd64 -t yourdockerhub/colqwen-embedding:latest .
docker push yourdockerhub/colqwen-embedding:latest
```

### 2. Create RunPod serverless endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new **Template**:
   - Image: `yourdockerhub/colqwen-embedding:latest`
   - Container disk: 20 GB
   - Environment variable: `EMBEDDING_API_KEY=your-secret-key`
3. Create a new **Endpoint** using that template:
   - GPU: L4 or A10G (24 GB VRAM minimum)
   - Min workers: 0 (scale to zero)
   - Max workers: 1–3
   - Idle timeout: 60s

### 3. Note your endpoint ID and RunPod API key

These are needed by client applications to call the endpoint.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EMBEDDING_API_KEY` | No | Custom API key for request validation (defense-in-depth) |

## Local Testing

```bash
pip install -r requirements.txt
python handler.py --rp_serve_api
```

Then send requests to `http://localhost:8000/runsync` with the same JSON format.
