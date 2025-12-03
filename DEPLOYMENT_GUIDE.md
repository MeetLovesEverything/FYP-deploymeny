# Render Deployment Fix Guide

## What Was Wrong

Your FastAPI application was failing to deploy on Render because:
1. The application wasn't binding to the port properly
2. There was insufficient error handling during startup
3. No health check endpoint for Render to verify the service is running
4. Missing detailed logging to debug deployment issues

## What I Fixed

### 1. **Enhanced Error Handling in `main.py`**
   - Added comprehensive logging during startup
   - Made the app start even if the model fails to load
   - Added try-except blocks with detailed error messages
   - Added compatibility for different PyTorch versions

### 2. **Added Health Check Endpoint**
   - New `/health` endpoint for Render to verify the service is running
   - Enhanced `/` endpoint with more detailed status information

### 3. **Created `render.yaml`**
   - Explicit configuration for Render deployment
   - Specifies Python version and build/start commands

## Next Steps to Deploy

### Option 1: Automatic Deployment (If Auto-Deploy is Enabled)
If you have auto-deploy enabled on Render, your changes have already been pushed to GitHub and Render should automatically start a new deployment.

1. Go to your Render dashboard: https://dashboard.render.com/
2. Find your service
3. Check the "Events" tab to see the new deployment starting
4. Monitor the logs for the startup messages

### Option 2: Manual Deployment
If auto-deploy is not enabled:

1. Go to https://dashboard.render.com/
2. Select your service
3. Click "Manual Deploy" → "Deploy latest commit"
4. Monitor the deployment logs

## What to Look For in Render Logs

After deployment, you should see these messages in the logs:

```
==================================================
Starting application...
Current working directory: /opt/render/project/src
Files in directory: [...]
==================================================
Model architecture initialized
Loading model from net.pt...
✓ Model loaded successfully from net.pt
==================================================
Application startup complete
==================================================
```

Then you should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10000
```

## Testing Your Deployed API

Once deployed successfully, test these endpoints:

### 1. Health Check
```bash
curl https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Root Endpoint
```bash
curl https://your-app-name.onrender.com/
```

Expected response:
```json
{
  "message": "Genre Classification API is running",
  "status": "healthy",
  "model_loaded": true,
  "endpoints": {
    "predict": "/predict (POST)"
  }
}
```

### 3. Predict Endpoint (with audio file)
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -F "file=@your-audio-file.mp3"
```

## Common Issues and Solutions

### Issue 1: "Model not found" in logs
**Solution:** The `net.pt` file might not be in your Git repository.
- Check: `git ls-files | grep net.pt`
- If not listed, add it: `git add net.pt && git commit -m "Add model file" && git push`

### Issue 2: Still timing out
**Possible causes:**
1. **Large model file taking too long to load**
   - Consider using Git LFS for the model file
   - Or host the model externally (S3, Google Cloud Storage) and download it during startup

2. **Memory issues**
   - Upgrade your Render plan to get more RAM
   - The model + PyTorch + librosa can be memory-intensive

### Issue 3: Port binding errors
The updated code now uses `os.getenv("PORT", 8000)` which Render automatically sets.

## Using Git LFS for Large Model Files (If Needed)

If the model file is causing issues, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track the model file
git lfs track "*.pt"

# Add the .gitattributes file
git add .gitattributes

# Add and commit the model
git add net.pt
git commit -m "Track model with Git LFS"
git push
```

Then in Render, you may need to enable Git LFS in your service settings.

## Alternative: Download Model from External Storage

If Git LFS doesn't work, modify the startup function to download the model:

```python
@app.on_event("startup")
async def load_model():
    global model
    
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        import requests
        print("Downloading model from external storage...")
        model_url = "YOUR_MODEL_URL_HERE"  # e.g., from Google Drive, S3, etc.
        response = requests.get(model_url)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully")
    
    # Rest of the loading code...
```

## Monitoring

After deployment:
1. Check the Render logs regularly
2. Use the `/health` endpoint to monitor service status
3. Set up Render's health check feature to automatically restart if the service fails

## Support

If you continue to have issues:
1. Share the complete Render deployment logs
2. Check if the model file size is within Render's limits
3. Consider upgrading your Render plan for more resources
