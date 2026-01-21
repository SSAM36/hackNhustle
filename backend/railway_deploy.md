# Railway Deployment Fix for CORS Issues

## Issue
After deploying to Railway, you're getting CORS errors when trying to access the API.

## What Was Fixed

### 1. Enhanced CORS Configuration
Updated `app.py` to include explicit CORS settings:
- Allow all origins (for testing; restrict in production if needed)
- Allow all necessary HTTP methods
- Include Authorization headers
- Enable credentials support

### 2. Production-Ready Gunicorn Configuration
Updated `Procfile` with:
- Multiple workers (4) for better performance
- Increased timeout (120s) for long-running requests
- Proper binding to Railway's PORT

### 3. Environment-Aware Configuration
Added environment detection in `app.py`:
- Debug mode OFF in production
- Dynamic port configuration
- Environment-specific logging

## Required Railway Environment Variables

Set these in your Railway project dashboard:

```
ENVIRONMENT=production
MONGO_URI=your_mongodb_atlas_connection_string
JWT_SECRET_KEY=your_secret_key_here
```

## After Deployment

1. **Test the Health Endpoint**
   ```
   https://hacknhustle-production.up.railway.app/health
   ```
   You should get a JSON response with status information.

2. **Update Frontend CORS**
   If you have a specific frontend domain, you can restrict CORS origins by:
   - Setting `CORS_ORIGINS` environment variable in Railway
   - Modifying the CORS config in app.py to use this variable

3. **Check Railway Logs**
   ```bash
   railway logs
   ```
   Look for any startup errors or connection issues.

## Common Issues & Solutions

### Issue: Still seeing CORS errors
**Solution**: Clear browser cache and hard refresh (Ctrl+Shift+R)

### Issue: 503 Service Unavailable
**Solution**: Check Railway logs - likely MongoDB connection issue

### Issue: Endpoints returning 404
**Solution**: Verify the Railway URL is correct and deployment succeeded

## Frontend Configuration

Update your frontend API base URL to:
```javascript
const API_BASE_URL = 'https://hacknhustle-production.up.railway.app';
```

## Testing the Deployment

Use curl to test endpoints:

```bash
# Health check
curl https://hacknhustle-production.up.railway.app/health

# Test route
curl https://hacknhustle-production.up.railway.app/test

# Register user
curl -X POST https://hacknhustle-production.up.railway.app/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"password123"}'
```

## Redeploy to Railway

After making these changes:

```bash
# Commit changes
git add .
git commit -m "Fix CORS and production configuration"
git push

# Railway will auto-deploy if connected to GitHub
# Or use Railway CLI:
railway up
```

## Monitor Deployment

1. Check Railway dashboard for deployment status
2. View logs in real-time: `railway logs --follow`
3. Test all critical endpoints after deployment

## Security Notes

- The current CORS config allows all origins (`*`) - good for testing
- For production, restrict origins to your frontend domain only
- Always use HTTPS in production
- Keep JWT_SECRET_KEY secure and never commit to Git
